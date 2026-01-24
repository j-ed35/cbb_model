"""
Centralized model loading utilities.

This module provides a single source of truth for loading trained models
across the codebase (prediction, backtesting, ensemble evaluation).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def load_trained_models(models_dir: Path) -> dict[str, Any]:
    """
    Load all trained models from the models directory.

    Args:
        models_dir: Path to the reports/models directory

    Returns:
        Dictionary with loaded models:
            - ridge: dict with 'model', 'scaler', 'feature_cols'
            - gbm: dict with 'model', 'feature_cols'
            - dnn: dict with 'model', 'scalers', 'feature_groups'
    """
    models = {}

    # Load Ridge model
    ridge_path = models_dir / "enhanced_ridge.pkl"
    if ridge_path.exists():
        with open(ridge_path, "rb") as f:
            models["ridge"] = pickle.load(f)

    # Load GBM model
    gbm_path = models_dir / "enhanced_gbm.pkl"
    if gbm_path.exists():
        with open(gbm_path, "rb") as f:
            models["gbm"] = pickle.load(f)

    # Load DNN model
    dnn_pt_path = models_dir / "dnn_enhanced.pt"
    dnn_artifacts_path = models_dir / "dnn_enhanced_artifacts.pkl"

    if dnn_pt_path.exists() and dnn_artifacts_path.exists():
        import torch

        with open(dnn_artifacts_path, "rb") as f:
            dnn_artifacts = pickle.load(f)

        from src.cbb.models.dnn_enhanced import TwoTowerDNN

        config = dnn_artifacts["model_config"]
        model = TwoTowerDNN(
            team_feature_dim=config["team_feature_dim"],
            matchup_feature_dim=config["matchup_feature_dim"],
            context_feature_dim=config["context_feature_dim"],
        )
        model.load_state_dict(torch.load(dnn_pt_path, weights_only=True))
        model.eval()

        models["dnn"] = {
            "model": model,
            "scalers": dnn_artifacts["scalers"],
            "feature_groups": dnn_artifacts["feature_groups"],
        }

    return models


def predict_with_model(
    model_name: str,
    model_data: dict[str, Any],
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions from a single model.

    Args:
        model_name: Name of the model ('ridge', 'gbm', or 'dnn')
        model_data: Model data dictionary from load_trained_models
        df: DataFrame with feature columns

    Returns:
        Array of predictions
    """
    if model_name == "ridge":
        return _predict_ridge(model_data, df)
    elif model_name == "gbm":
        return _predict_gbm(model_data, df)
    elif model_name == "dnn":
        return _predict_dnn(model_data, df)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_ensemble(
    models: dict[str, Any],
    df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """
    Generate ensemble predictions from multiple models.

    Args:
        models: Dictionary of loaded models from load_trained_models
        df: DataFrame with feature columns
        weights: Optional custom weights (default: 0.1 ridge, 0.2 gbm, 0.7 dnn)

    Returns:
        Weighted ensemble predictions
    """
    if weights is None:
        weights = {"ridge": 0.1, "gbm": 0.2, "dnn": 0.7}

    # Normalize weights to available models
    available = [name for name in weights if name in models]
    total_weight = sum(weights[name] for name in available)
    normalized_weights = {name: weights[name] / total_weight for name in available}

    # Generate predictions
    predictions = {}
    for name in available:
        predictions[name] = predict_with_model(name, models[name], df)

    # Combine
    ensemble = np.zeros(len(df))
    for name, w in normalized_weights.items():
        ensemble += w * predictions[name]

    return ensemble


def _predict_ridge(model_data: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from Ridge model."""
    feature_cols = model_data["feature_cols"]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    X_scaled = model_data["scaler"].transform(X)
    return model_data["model"].predict(X_scaled)


def _predict_gbm(model_data: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from GBM model."""
    feature_cols = model_data["feature_cols"]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    return model_data["model"].predict(X)


def _predict_dnn(model_data: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from DNN model."""
    import torch

    scalers = model_data["scalers"]
    feature_groups = model_data["feature_groups"]
    model = model_data["model"]

    def get_features(feature_list: list[str]) -> np.ndarray:
        available = [f for f in feature_list if f in df.columns]
        data = df[available].fillna(0).values
        return np.nan_to_num(data, nan=0)

    team_a = scalers["team_a"].transform(get_features(feature_groups["team_a"]))
    team_b = scalers["team_b"].transform(get_features(feature_groups["team_b"]))
    matchup = scalers["matchup"].transform(get_features(feature_groups["matchup"]))
    context = scalers["context"].transform(get_features(feature_groups["context"]))

    team_a_t = torch.FloatTensor(team_a)
    team_b_t = torch.FloatTensor(team_b)
    matchup_t = torch.FloatTensor(matchup)
    context_t = torch.FloatTensor(context)

    with torch.no_grad():
        preds, _ = model(team_a_t, team_b_t, matchup_t, context_t)

    return preds.numpy()


def get_model_feature_columns(models: dict[str, Any]) -> list[str]:
    """
    Get the union of all feature columns used by the models.

    Args:
        models: Dictionary of loaded models

    Returns:
        List of unique feature column names
    """
    all_cols = set()

    if "ridge" in models:
        all_cols.update(models["ridge"].get("feature_cols", []))

    if "gbm" in models:
        all_cols.update(models["gbm"].get("feature_cols", []))

    if "dnn" in models:
        feature_groups = models["dnn"].get("feature_groups", {})
        for group_cols in feature_groups.values():
            all_cols.update(group_cols)

    return sorted(all_cols)


def load_features_for_prediction(
    features_path: Path,
    models: dict[str, Any],
) -> pd.DataFrame:
    """
    Load feature file with only columns required for prediction.

    This is more memory-efficient than loading the full parquet file,
    especially for large datasets.

    Args:
        features_path: Path to the parquet file
        models: Dictionary of loaded models (to determine required columns)

    Returns:
        DataFrame with only required columns
    """
    # Get columns needed for prediction
    model_cols = get_model_feature_columns(models)

    # Always include these metadata columns for filtering/splitting
    required_cols = [
        "game_key",
        "season",
        "date",
        "team_a",
        "team_b",
        "spread_a",
        "final_margin_a",
        "cover_a",
        "kp_matched",
    ]

    # Combine and deduplicate
    all_cols = list(set(required_cols + model_cols))

    # Load with column selection
    return pd.read_parquet(features_path, columns=all_cols)

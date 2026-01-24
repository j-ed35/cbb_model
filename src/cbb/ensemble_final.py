"""
Final ensemble model combining Ridge, GBM, and DNN predictions.

This script:
1. Loads the three trained models
2. Creates weighted ensemble predictions
3. Tunes weights on validation set
4. Reports final test performance

Target: >52.4% hit rate (breakeven for -110 odds)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

from src.cbb.utils.evaluation import evaluate_ats
from src.cbb.utils.model_loader import (
    load_trained_models,
    predict_with_model,
    load_features_for_prediction,
)


def evaluate_ensemble(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
    spreads: np.ndarray,
    covers: np.ndarray,
) -> dict[str, Any]:
    """Evaluate weighted ensemble."""
    # Combine predictions
    ensemble_pred = np.zeros_like(list(predictions.values())[0])
    for name, weight in weights.items():
        if name in predictions:
            ensemble_pred += weight * predictions[name]

    # Use shared evaluation utility
    result = evaluate_ats(
        predictions=ensemble_pred,
        spreads=spreads,
        covers=covers,
        threshold=0,
    )

    # Add ensemble predictions to result
    result["predictions"] = ensemble_pred

    return result


def tune_weights(
    predictions: dict[str, np.ndarray],
    spreads: np.ndarray,
    covers: np.ndarray,
) -> dict[str, float]:
    """Tune ensemble weights to maximize hit rate."""
    model_names = list(predictions.keys())
    n_models = len(model_names)

    def objective(w: np.ndarray) -> float:
        # Ensure weights sum to 1
        w = w / w.sum()
        weights = {name: w[i] for i, name in enumerate(model_names)}
        result = evaluate_ensemble(predictions, weights, spreads, covers)
        # Maximize hit rate (minimize negative)
        return -result["hit_rate"]

    # Start with equal weights
    w0 = np.ones(n_models) / n_models

    # Constraints: weights must be positive
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    # Optimize
    result = minimize(objective, w0, method="SLSQP", bounds=bounds)

    # Normalize
    optimal_w = result.x / result.x.sum()

    return {name: float(optimal_w[i]) for i, name in enumerate(model_names)}


def main() -> None:
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    features_dir = project_root / "data" / "features"
    models_dir = project_root / "reports" / "models"
    metrics_dir = project_root / "reports" / "metrics"

    # Load models first to determine required columns
    print("Loading models...")
    models = load_trained_models(models_dir)

    if len(models) == 0:
        print("No models found! Train models first.")
        return

    for name in models:
        print(f"  Loaded {name}")

    # Load features with only required columns (memory optimization)
    print("\nLoading features...")
    features_df = load_features_for_prediction(
        features_dir / "games_features_enhanced.parquet",
        models,
    )

    # Filter to usable games
    df = features_df[
        (features_df["kp_matched"] == True) & (features_df["spread_a"].notna())
    ].copy()
    df = df.dropna(subset=["final_margin_a", "cover_a"])

    print(f"Loaded {len(df):,} usable games")

    # Split by season
    train_df = df[df["season"].isin(["2022-23", "2023-24"])]
    val_df = df[df["season"] == "2024-25"]
    test_df = df[df["season"] == "2025-26"]

    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    print(f"Test:  {len(test_df):,}")

    # Generate predictions for each split
    print("\nGenerating predictions...")

    splits = {
        "val": val_df,
        "test": test_df,
    }

    results: dict = {}

    for split_name, split_df in splits.items():
        predictions = {}

        for model_name in models:
            predictions[model_name] = predict_with_model(
                model_name, models[model_name], split_df
            )
            print(f"  {split_name} {model_name} predictions: {len(predictions[model_name])}")

        results[split_name] = {
            "predictions": predictions,
            "spreads": split_df["spread_a"].values,
            "covers": split_df["cover_a"].values,
            "margins": split_df["final_margin_a"].values,
        }

    # Tune weights on validation set
    print("\n" + "=" * 60)
    print("TUNING ENSEMBLE WEIGHTS ON VALIDATION SET")
    print("=" * 60)

    optimal_weights = tune_weights(
        results["val"]["predictions"],
        results["val"]["spreads"],
        results["val"]["covers"],
    )

    print("\nOptimal weights:")
    for name, weight in optimal_weights.items():
        print(f"  {name}: {weight:.3f}")

    # Evaluate on validation with optimal weights
    val_result = evaluate_ensemble(
        results["val"]["predictions"],
        optimal_weights,
        results["val"]["spreads"],
        results["val"]["covers"],
    )

    print(f"\nValidation ensemble performance:")
    print(f"  Hit Rate: {val_result['hit_rate']:.3f}")
    print(f"  ROI: {val_result['roi']:.3f}")

    # Evaluate on test with optimal weights
    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)

    test_result = evaluate_ensemble(
        results["test"]["predictions"],
        optimal_weights,
        results["test"]["spreads"],
        results["test"]["covers"],
    )

    print(f"\nTest ensemble performance:")
    print(f"  Hit Rate: {test_result['hit_rate']:.3f}")
    print(f"  ROI: {test_result['roi']:.3f}")
    print(f"  N bets: {test_result['n_bets']}")

    # Compare individual models on test
    print("\nIndividual model performance on test:")
    for name in results["test"]["predictions"].keys():
        individual = evaluate_ensemble(
            {name: results["test"]["predictions"][name]},
            {name: 1.0},
            results["test"]["spreads"],
            results["test"]["covers"],
        )
        print(f"  {name}: {individual['hit_rate']:.3f} hit rate, {individual['roi']:.3f} ROI")

    # Breakeven check
    breakeven = 0.524
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"Ensemble Hit Rate: {test_result['hit_rate']:.1%}")
    print(f"Breakeven:         {breakeven:.1%}")

    if test_result["hit_rate"] >= breakeven:
        print(f"\n PROFITABLE! {test_result['hit_rate'] - breakeven:.1%} above breakeven")
    else:
        print(f"\n {breakeven - test_result['hit_rate']:.1%} below breakeven")

    # Save results
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "ensemble_hit_rate": test_result["hit_rate"],
        "ensemble_roi": test_result["roi"],
        "n_bets": test_result["n_bets"],
        **{f"weight_{k}": v for k, v in optimal_weights.items()},
    }

    pd.DataFrame([summary]).to_csv(metrics_dir / "ensemble_final_summary.csv", index=False)
    print(f"\n Saved results to {metrics_dir / 'ensemble_final_summary.csv'}")

    # Save ensemble config
    with open(models_dir / "ensemble_final_config.pkl", "wb") as f:
        pickle.dump({
            "weights": optimal_weights,
        }, f)


if __name__ == "__main__":
    main()

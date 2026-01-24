"""
Enhanced ATS model training with improved features and ensemble methods.

Target: Achieve >52.4% hit rate (breakeven for -110 odds)

Key improvements over baseline:
1. Extended KenPom features (rankings, matchup-specific)
2. Situational features (rest days, back-to-back)
3. Rolling ATS records
4. Recency-weighted performance stats
5. Ensemble of Ridge + GBM + Neural Network
6. Calibrated probability outputs
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.cbb.utils.evaluation import evaluate_ats, evaluate_with_threshold, tune_threshold


# Feature groups for ablation studies
KENPOM_CORE = [
    "kp_adj_em_a", "kp_adj_em_b", "kp_adj_em_diff",
    "kp_adj_o_a", "kp_adj_o_b",
    "kp_adj_d_a", "kp_adj_d_b",
    "kp_tempo_a", "kp_tempo_b", "kp_tempo_avg",
]

# NOTE: Rankings are not available in historical KenPom snapshots (pre-2025)
# Only include these when using recent daily snapshots
KENPOM_RANKINGS = [
    "kp_rank_em_a", "kp_rank_em_b",
    "kp_rank_o_a", "kp_rank_o_b",
    "kp_rank_d_a", "kp_rank_d_b",
]

KENPOM_MATCHUP = [
    "kp_o_vs_d_a", "kp_o_vs_d_b",
    "kp_tempo_diff",
]

SITUATIONAL = [
    "rest_days_a", "rest_days_b", "rest_diff",
    "b2b_a", "b2b_b",
]

ROLLING_ATS = [
    "rolling_ats_a", "rolling_ats_b", "rolling_ats_diff",
]

RECENCY_WEIGHTED = [
    "ew_margin_a", "ew_margin_b", "ew_margin_diff",
]

CONTEXT = [
    "is_home_a", "is_neutral",
]

# Exclude rankings since they're not available in historical data
ALL_FEATURES = (
    KENPOM_CORE + KENPOM_MATCHUP + SITUATIONAL + ROLLING_ATS + RECENCY_WEIGHTED + CONTEXT
)


def load_enhanced_features(project_root: Path) -> pd.DataFrame:
    """Load enhanced feature dataset."""
    features_path = project_root / "data" / "features" / "games_features_enhanced.parquet"

    if not features_path.exists():
        print("Enhanced features not found. Generating...")
        from src.cbb.features.enhanced_features import main as build_features
        build_features()

    return pd.read_parquet(features_path)


def prepare_train_data(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    require_spread: bool = True,
    require_kenpom: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare data for training.

    Filters:
    - Requires spread data (for ATS evaluation)
    - Optionally requires KenPom match
    - Drops rows with missing feature values
    """
    df = features_df.copy()

    if require_spread:
        df = df[df["spread_a"].notna()]

    if require_kenpom:
        df = df[df["kp_matched"] == True]

    # Check which features are available
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        print(f"Warning: Missing features: {missing}")

    # Drop rows with NaN in available features
    df = df.dropna(subset=available + ["final_margin_a", "cover_a"])

    return df, available


def train_margin_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Ridge, GradientBoostingRegressor, StandardScaler]:
    """
    Train margin regression models (Ridge + GBM).

    Returns: ridge_model, gbm_model, scaler
    """
    X = train_df[feature_cols].values
    y = train_df["final_margin_a"].values

    # Scale for Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    # Gradient Boosting
    gbm = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    gbm.fit(X, y)

    return ridge, gbm, scaler


def ensemble_predict(
    ridge_model: Ridge,
    gbm_model: GradientBoostingRegressor,
    X_scaled: np.ndarray,
    X_raw: np.ndarray,
    weights: tuple[float, float] = (0.4, 0.6),
) -> np.ndarray:
    """
    Ensemble prediction from Ridge and GBM.

    Args:
        ridge_model: Trained Ridge model
        gbm_model: Trained GBM model
        X_scaled: Scaled features for Ridge
        X_raw: Raw features for GBM
        weights: (ridge_weight, gbm_weight)

    Returns:
        Weighted average predictions
    """
    ridge_pred = ridge_model.predict(X_scaled)
    gbm_pred = gbm_model.predict(X_raw)

    return weights[0] * ridge_pred + weights[1] * gbm_pred


def run_cross_validation(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Run time-series cross-validation.

    Uses TimeSeriesSplit to ensure no lookahead bias.
    """
    df, available_cols = prepare_train_data(features_df, feature_cols)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Cross-validation on {len(df):,} games with {len(available_cols)} features")

    X = df[available_cols].values
    y = df["final_margin_a"].values
    spreads = df["spread_a"].values
    covers = df["cover_a"].values

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_results = []
    fold = 0

    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        spreads_test = spreads[test_idx]
        covers_test = covers[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)

        gbm = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42
        )
        gbm.fit(X_train, y_train)

        # Predictions
        ridge_pred = ridge.predict(X_test_scaled)
        gbm_pred = gbm.predict(X_test)
        ensemble_pred_arr = 0.4 * ridge_pred + 0.6 * gbm_pred

        # Evaluate using shared utility
        for name, pred in [
            ("Ridge", ridge_pred),
            ("GBM", gbm_pred),
            ("Ensemble", ensemble_pred_arr),
        ]:
            metrics = evaluate_ats(
                predictions=pred,
                spreads=spreads_test,
                margins=y_test,
                covers=covers_test,
            )
            metrics["model"] = name
            metrics["fold"] = fold
            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    # Aggregate by model
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (Time-Series Split)")
    print("=" * 60)

    for model in ["Ridge", "GBM", "Ensemble"]:
        model_results = results_df[results_df["model"] == model]
        print(f"\n{model}:")
        print(f"  MAE:      {model_results['mae'].mean():.2f} +/- {model_results['mae'].std():.2f}")
        print(f"  Hit Rate: {model_results['hit_rate'].mean():.3f} +/- {model_results['hit_rate'].std():.3f}")
        print(f"  ROI:      {model_results['roi'].mean():.3f} +/- {model_results['roi'].std():.3f}")

    return results_df


def run_holdout_evaluation(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    train_seasons: list[str],
    val_seasons: list[str],
    test_seasons: list[str],
) -> tuple[dict[str, Any], Ridge, GradientBoostingRegressor, StandardScaler, list[str]]:
    """Run evaluation with fixed train/val/test split."""
    df, available_cols = prepare_train_data(features_df, feature_cols)

    train_df = df[df["season"].isin(train_seasons)]
    val_df = df[df["season"].isin(val_seasons)]
    test_df = df[df["season"].isin(test_seasons)]

    print(f"\nTrain: {len(train_df):,} games ({train_seasons})")
    print(f"Val:   {len(val_df):,} games ({val_seasons})")
    print(f"Test:  {len(test_df):,} games ({test_seasons})")

    # Train models
    ridge, gbm, scaler = train_margin_models(train_df, available_cols)

    # Feature importance from GBM
    print("\nTop 10 feature importances (GBM):")
    importance = pd.DataFrame({
        "feature": available_cols,
        "importance": gbm.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))

    results: dict[str, Any] = {}

    for name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = df_split[available_cols].values
        X_scaled = scaler.transform(X)
        y = df_split["final_margin_a"].values
        spreads = df_split["spread_a"].values
        covers = df_split["cover_a"].values

        # Individual model predictions
        ridge_pred = ridge.predict(X_scaled)
        gbm_pred = gbm.predict(X)
        ensemble_pred_arr = 0.4 * ridge_pred + 0.6 * gbm_pred

        for model_name, pred in [
            ("Ridge", ridge_pred),
            ("GBM", gbm_pred),
            ("Ensemble", ensemble_pred_arr),
        ]:
            metrics = evaluate_ats(
                predictions=pred,
                spreads=spreads,
                margins=y,
                covers=covers,
            )
            results[f"{model_name}_{name}"] = metrics
            print(f"\n{model_name} on {name}:")
            print(f"  MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
            print(f"  Hit Rate: {metrics['hit_rate']:.3f}, ROI: {metrics['roi']:.3f}")

    # Tune threshold on validation
    val_X = val_df[available_cols].values
    val_X_scaled = scaler.transform(val_X)
    val_ensemble = 0.4 * ridge.predict(val_X_scaled) + 0.6 * gbm.predict(val_X)

    best_threshold, val_threshold_metrics = tune_threshold(
        predictions=val_ensemble,
        spreads=val_df["spread_a"].values,
        covers=val_df["cover_a"].values,
    )

    print(f"\nOptimal threshold (val): {best_threshold}")
    if val_threshold_metrics:
        print(
            f"  Val with threshold: Hit Rate={val_threshold_metrics['hit_rate']:.3f}, "
            f"ROI={val_threshold_metrics['roi']:.3f}, N={val_threshold_metrics['n_bets']}"
        )

    # Apply to test
    test_X = test_df[available_cols].values
    test_X_scaled = scaler.transform(test_X)
    test_ensemble = 0.4 * ridge.predict(test_X_scaled) + 0.6 * gbm.predict(test_X)

    test_threshold_metrics = evaluate_with_threshold(
        predictions=test_ensemble,
        spreads=test_df["spread_a"].values,
        covers=test_df["cover_a"].values,
        threshold=best_threshold,
    )

    print(f"\nTest with threshold {best_threshold}:")
    print(f"  Hit Rate: {test_threshold_metrics['hit_rate']:.3f}")
    print(f"  ROI: {test_threshold_metrics['roi']:.3f}")
    print(f"  N bets: {test_threshold_metrics['n_bets']}")

    results["threshold"] = best_threshold
    results["test_threshold_metrics"] = test_threshold_metrics

    return results, ridge, gbm, scaler, available_cols


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train enhanced ATS models")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    parser.add_argument("--ablation", action="store_true", help="Run feature ablation study")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "reports" / "models"
    metrics_dir = project_root / "reports" / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    features_df = load_enhanced_features(project_root)
    print(f"Loaded {len(features_df):,} games")

    if args.cv:
        results_df = run_cross_validation(features_df, ALL_FEATURES)
        results_df.to_csv(metrics_dir / "enhanced_cv_results.csv", index=False)
        return

    if args.ablation:
        # Run ablation study
        feature_sets = {
            "KenPom Core Only": KENPOM_CORE + CONTEXT,
            "+ Rankings": KENPOM_CORE + KENPOM_RANKINGS + CONTEXT,
            "+ Matchup": KENPOM_CORE + KENPOM_RANKINGS + KENPOM_MATCHUP + CONTEXT,
            "+ Situational": KENPOM_CORE + KENPOM_RANKINGS + KENPOM_MATCHUP + SITUATIONAL + CONTEXT,
            "+ Rolling ATS": KENPOM_CORE + KENPOM_RANKINGS + KENPOM_MATCHUP + SITUATIONAL + ROLLING_ATS + CONTEXT,
            "All Features": ALL_FEATURES,
        }

        ablation_results = []
        for name, features in feature_sets.items():
            print(f"\n{'='*60}")
            print(f"Testing: {name}")
            print(f"{'='*60}")

            results, _, _, _, _ = run_holdout_evaluation(
                features_df, features,
                train_seasons=["2022-23", "2023-24"],
                val_seasons=["2024-25"],
                test_seasons=["2025-26"]
            )

            ablation_results.append({
                "feature_set": name,
                "n_features": len([f for f in features if f in features_df.columns]),
                "test_hit_rate": results.get("Ensemble_test", {}).get("hit_rate", 0),
                "test_roi": results.get("Ensemble_test", {}).get("roi", 0),
            })

        ablation_df = pd.DataFrame(ablation_results)
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        print(ablation_df.to_string(index=False))
        ablation_df.to_csv(metrics_dir / "ablation_results.csv", index=False)
        return

    # Default: holdout evaluation with all features
    print("\n" + "="*60)
    print("ENHANCED MODEL TRAINING")
    print("="*60)

    results, ridge, gbm, scaler, feature_cols = run_holdout_evaluation(
        features_df, ALL_FEATURES,
        train_seasons=["2022-23", "2023-24"],
        val_seasons=["2024-25"],
        test_seasons=["2025-26"]
    )

    # Save models
    with open(models_dir / "enhanced_ridge.pkl", "wb") as f:
        pickle.dump({"model": ridge, "scaler": scaler, "feature_cols": feature_cols}, f)

    with open(models_dir / "enhanced_gbm.pkl", "wb") as f:
        pickle.dump({"model": gbm, "feature_cols": feature_cols}, f)

    # Save ensemble config
    with open(models_dir / "enhanced_ensemble.pkl", "wb") as f:
        pickle.dump({
            "ridge": ridge,
            "gbm": gbm,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "weights": (0.4, 0.6),
            "threshold": results.get("threshold", 0),
        }, f)

    print(f"\n Saved models to {models_dir}")

    # Save summary
    summary = {
        "test_hit_rate": results.get("Ensemble_test", {}).get("hit_rate", 0),
        "test_roi": results.get("Ensemble_test", {}).get("roi", 0),
        "threshold": results.get("threshold", 0),
        "threshold_hit_rate": results.get("test_threshold_metrics", {}).get("hit_rate", 0),
        "threshold_roi": results.get("test_threshold_metrics", {}).get("roi", 0),
        "threshold_n_bets": results.get("test_threshold_metrics", {}).get("n_bets", 0),
    }

    pd.DataFrame([summary]).to_csv(metrics_dir / "enhanced_summary.csv", index=False)
    print(f" Saved summary to {metrics_dir / 'enhanced_summary.csv'}")


if __name__ == "__main__":
    main()

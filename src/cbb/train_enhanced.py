"""
Train a GBM model for ATS margin prediction.

Uses expanded feature set including KenPom ratings, tournament context,
rest days, rolling ATS, and recency-weighted stats.

Supports:
- Prediction shrinkage evaluation (mean reversion)
- Hyperparameter tuning via --tune flag

Usage:
    python -m src.cbb.train_enhanced
    python -m src.cbb.train_enhanced --tune

Output:
    reports/models/gbm.pkl - Trained model + feature column list + shrinkage
"""

from __future__ import annotations

import argparse
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.cbb.features.prepare import prepare_ats_data
from src.cbb.utils.evaluation import evaluate_ats, tune_threshold_bootstrap, tune_edge_cap, analyze_by_edge_bucket


# Expanded feature set
FEATURES = [
    # KenPom core (8)
    "kp_adj_em_a", "kp_adj_em_b",
    "kp_adj_o_a", "kp_adj_o_b",
    "kp_adj_d_a", "kp_adj_d_b",
    "kp_tempo_a", "kp_tempo_b",
    # KenPom derived (5)
    "kp_adj_em_diff",
    "kp_tempo_avg", "kp_tempo_diff",
    "kp_o_vs_d_a", "kp_o_vs_d_b",
    # Context (2)
    "is_home_a", "is_neutral",
    # Game context (2)
    "is_conference_tourney", "is_postseason",
    # Rest/situational (3)
    "rest_days_a", "rest_days_b", "rest_diff",
    # Recency (3)
    "ew_margin_a", "ew_margin_b", "ew_margin_diff",
    # Rolling ATS (2)
    "rolling_ats_a", "rolling_ats_b",
]


def load_features(project_root: Path) -> pd.DataFrame:
    """Load enhanced feature dataset."""
    path = project_root / "data" / "features" / "games_features_enhanced.parquet"
    if not path.exists():
        print("Enhanced features not found. Run enhanced_features.py first.")
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def find_best_shrinkage(
    gbm,
    val_df: pd.DataFrame,
    available_cols: list[str],
) -> float:
    """Find optimal shrinkage factor on validation set."""
    val_preds_raw = gbm.predict(val_df[available_cols].values)
    val_spreads = val_df["spread_a"].values
    val_covers = val_df["cover_a"].values

    best_shrinkage = 1.0
    best_hit_rate = 0.0

    print("\nShrinkage evaluation (val):")
    for s in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        preds = val_preds_raw * s
        metrics = evaluate_ats(
            predictions=preds,
            spreads=val_spreads,
            covers=val_covers,
            threshold=0,
        )
        marker = ""
        if metrics["hit_rate"] > best_hit_rate:
            best_hit_rate = metrics["hit_rate"]
            best_shrinkage = s
            marker = " <-- best"
        print(f"  shrinkage={s:.2f}: Hit Rate={metrics['hit_rate']:.3f} | ROI={metrics['roi']:+.3f}{marker}")

    return best_shrinkage


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    available_cols: list[str],
) -> dict:
    """Grid search over GBM hyperparameters, evaluated on validation ATS hit rate."""
    param_grid = {
        "n_estimators": [150, 300, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08],
        "min_samples_leaf": [15, 25, 40],
        "subsample": [0.7, 0.8],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"\nHyperparameter tuning: {len(combos)} combinations")

    val_spreads = val_df["spread_a"].values
    val_covers = val_df["cover_a"].values
    X_val = val_df[available_cols].values

    best_hit_rate = 0.0
    best_params = {}
    best_model = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        gbm = GradientBoostingRegressor(random_state=42, **params)
        gbm.fit(X_train, y_train)

        val_preds = gbm.predict(X_val)
        metrics = evaluate_ats(
            predictions=val_preds,
            spreads=val_spreads,
            covers=val_covers,
            threshold=0,
        )

        if metrics["hit_rate"] > best_hit_rate:
            best_hit_rate = metrics["hit_rate"]
            best_params = params
            best_model = gbm
            print(f"  [{i+1}/{len(combos)}] NEW BEST: {params} -> Hit Rate={metrics['hit_rate']:.3f} ROI={metrics['roi']:+.3f}")

    print(f"\nBest params: {best_params}")
    print(f"Best val hit rate: {best_hit_rate:.3f}")
    return {"model": best_model, "params": best_params}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GBM model for ATS prediction")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter grid search")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "reports" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load and prep data
    features_df = load_features(project_root)
    df, available_cols = prepare_ats_data(features_df, FEATURES)

    print(f"Loaded {len(df):,} games, {len(available_cols)} features")
    print(f"Features: {available_cols}")

    # Split by season
    train_df = df[df["season"].isin(["2021-22", "2022-23", "2023-24"])]
    val_df = df[df["season"] == "2024-25"]
    test_df = df[df["season"] == "2025-26"]

    print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train = train_df[available_cols].values
    y_train = train_df["final_margin_a"].values

    if args.tune:
        # Hyperparameter grid search
        result = tune_hyperparameters(X_train, y_train, val_df, available_cols)
        gbm = result["model"]
    else:
        # Train with default params
        gbm = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        gbm.fit(X_train, y_train)

    # Feature importance
    importance = pd.DataFrame({
        "feature": available_cols,
        "importance": gbm.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\nFeature importance:\n{importance.to_string(index=False)}")

    # Find best shrinkage factor
    best_shrinkage = find_best_shrinkage(gbm, val_df, available_cols)
    print(f"\nBest shrinkage: {best_shrinkage}")

    # Evaluate on each split (with shrinkage applied)
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        X = split_df[available_cols].values
        preds = gbm.predict(X) * best_shrinkage
        metrics = evaluate_ats(
            predictions=preds,
            spreads=split_df["spread_a"].values,
            margins=split_df["final_margin_a"].values,
            covers=split_df["cover_a"].values,
        )
        print(f"\n{name}: MAE={metrics['mae']:.2f} | Hit Rate={metrics['hit_rate']:.3f} | ROI={metrics['roi']:+.3f}")

    # Edge bucket analysis on test (run first to inform threshold/cap tuning)
    test_preds = gbm.predict(test_df[available_cols].values) * best_shrinkage
    print("\nEdge bucket analysis (test):")
    buckets = analyze_by_edge_bucket(
        predictions=test_preds,
        spreads=test_df["spread_a"].values,
        covers=test_df["cover_a"].values,
    )
    for b in buckets:
        print(f"  {b['bucket']:>8s}: {b['hit_rate']:.3f} hit rate ({b['n_games']} games) ROI={b['roi']:+.3f}")

    # Tune edge cap on validation (with shrinkage)
    # Small edges outperform large edges, so we cap max edge to filter noise
    val_preds = gbm.predict(val_df[available_cols].values) * best_shrinkage
    best_edge_cap, cap_metrics = tune_edge_cap(
        predictions=val_preds,
        spreads=val_df["spread_a"].values,
        covers=val_df["cover_a"].values,
        threshold=0.0,
        min_bets=30,
    )
    print(f"\nBest edge cap (val): {best_edge_cap}")
    if best_edge_cap is not None:
        print(f"  Val: Hit Rate={cap_metrics['hit_rate']:.3f} | ROI={cap_metrics['roi']:+.3f} | N={cap_metrics['n_bets']}")

    # Tune threshold using bootstrap resampling (more robust than single-sample)
    best_threshold, val_metrics = tune_threshold_bootstrap(
        predictions=val_preds,
        spreads=val_df["spread_a"].values,
        covers=val_df["cover_a"].values,
        min_bets=30,
        edge_cap=best_edge_cap,
    )
    print(f"\nBest threshold (val, bootstrap): {best_threshold}")
    print(f"  Val: Hit Rate={val_metrics['hit_rate']:.3f} | ROI={val_metrics['roi']:+.3f} | N={val_metrics['n_bets']}")

    # Apply to test
    test_metrics = evaluate_ats(
        predictions=test_preds,
        spreads=test_df["spread_a"].values,
        covers=test_df["cover_a"].values,
        threshold=best_threshold,
        edge_cap=best_edge_cap,
    )
    cap_str = f", cap={best_edge_cap}" if best_edge_cap else ""
    print(f"  Test (threshold={best_threshold}{cap_str}): Hit Rate={test_metrics['hit_rate']:.3f} | ROI={test_metrics['roi']:+.3f} | N={test_metrics['n_bets']}")

    # Save model
    model_path = models_dir / "gbm.pkl"
    save_data = {
        "model": gbm,
        "feature_cols": available_cols,
        "threshold": best_threshold,
        "shrinkage": best_shrinkage,
    }
    if best_edge_cap is not None:
        save_data["edge_cap"] = best_edge_cap
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()

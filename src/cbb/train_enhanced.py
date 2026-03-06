"""
Train a single GBM model for ATS margin prediction.

Uses only features available at live prediction time (KenPom ratings + context).
No rolling stats, no rest days - those default to neutral at inference anyway.

Usage:
    python -m src.cbb.train_enhanced

Output:
    reports/models/gbm.pkl - Trained model + feature column list
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.cbb.features.prepare import prepare_ats_data
from src.cbb.utils.evaluation import evaluate_ats, tune_threshold, analyze_by_edge_bucket


# Only features available at live prediction time
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
]


def load_features(project_root: Path) -> pd.DataFrame:
    """Load enhanced feature dataset."""
    path = project_root / "data" / "features" / "games_features_enhanced.parquet"
    if not path.exists():
        print("Enhanced features not found. Run enhanced_features.py first.")
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def main() -> None:
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "reports" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load and prep data
    features_df = load_features(project_root)
    df, available_cols = prepare_ats_data(features_df, FEATURES)

    print(f"Loaded {len(df):,} games, {len(available_cols)} features")
    print(f"Features: {available_cols}")

    # Split by season (3 seasons train for more data)
    train_df = df[df["season"].isin(["2021-22", "2022-23", "2023-24"])]
    val_df = df[df["season"] == "2024-25"]
    test_df = df[df["season"] == "2025-26"]

    print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Train GBM
    X_train = train_df[available_cols].values
    y_train = train_df["final_margin_a"].values

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

    # Evaluate on each split
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        X = split_df[available_cols].values
        preds = gbm.predict(X)
        metrics = evaluate_ats(
            predictions=preds,
            spreads=split_df["spread_a"].values,
            margins=split_df["final_margin_a"].values,
            covers=split_df["cover_a"].values,
        )
        print(f"\n{name}: MAE={metrics['mae']:.2f} | Hit Rate={metrics['hit_rate']:.3f} | ROI={metrics['roi']:+.3f}")

    # Tune threshold on validation
    val_preds = gbm.predict(val_df[available_cols].values)
    best_threshold, val_metrics = tune_threshold(
        predictions=val_preds,
        spreads=val_df["spread_a"].values,
        covers=val_df["cover_a"].values,
    )
    print(f"\nBest threshold (val): {best_threshold}")
    print(f"  Val: Hit Rate={val_metrics['hit_rate']:.3f} | ROI={val_metrics['roi']:+.3f} | N={val_metrics['n_bets']}")

    # Apply to test
    test_preds = gbm.predict(test_df[available_cols].values)
    test_metrics = evaluate_ats(
        predictions=test_preds,
        spreads=test_df["spread_a"].values,
        covers=test_df["cover_a"].values,
        threshold=best_threshold,
    )
    print(f"  Test: Hit Rate={test_metrics['hit_rate']:.3f} | ROI={test_metrics['roi']:+.3f} | N={test_metrics['n_bets']}")

    # Edge bucket analysis on test
    print("\nEdge bucket analysis (test):")
    buckets = analyze_by_edge_bucket(
        predictions=test_preds,
        spreads=test_df["spread_a"].values,
        covers=test_df["cover_a"].values,
    )
    for b in buckets:
        print(f"  {b['bucket']:>8s}: {b['hit_rate']:.3f} hit rate ({b['n_games']} games) ROI={b['roi']:+.3f}")

    # Save model
    model_path = models_dir / "gbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": gbm,
            "feature_cols": available_cols,
            "threshold": best_threshold,
        }, f)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()

"""
Selective betting backtest with edge thresholds.

Key insight: Instead of betting every game, we only bet when the model
shows a strong edge. This trades volume for accuracy.

This script:
1. Loads the trained models and generates predictions
2. Tests various edge thresholds to find optimal selectivity
3. Analyzes performance by confidence buckets
4. Reports final test performance with optimal threshold
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.cbb.utils.evaluation import (
    evaluate_ats,
    evaluate_with_threshold,
    analyze_by_edge_bucket,
)
from src.cbb.utils.model_loader import (
    load_trained_models,
    predict_with_model,
    predict_ensemble,
    load_features_for_prediction,
)


def analyze_model_agreement(
    predictions: dict[str, np.ndarray],
    spreads: np.ndarray,
    covers: np.ndarray,
) -> pd.DataFrame:
    """Analyze performance when models agree vs disagree."""
    # Calculate edges for each model
    edges = {}
    for name, preds in predictions.items():
        edges[name] = preds - (-spreads)

    # Filter valid
    valid_mask = np.isin(covers, [0, 1])
    valid_covers = covers[valid_mask]

    # Count agreements (all models predict same side)
    signs = np.array([np.sign(edges[name][valid_mask]) for name in edges.keys()])
    all_agree = (signs == signs[0]).all(axis=0)

    results = []

    # Helper for ROI
    def compute_roi(hits: np.ndarray) -> float:
        if len(hits) == 0:
            return 0.0
        win_payout = 100 / 110
        return (hits.sum() * win_payout - (~hits).sum()) / len(hits)

    # When all models agree
    if all_agree.sum() > 0:
        agree_edge = np.mean(
            [edges[name][valid_mask][all_agree] for name in edges.keys()], axis=0
        )
        agree_covers = valid_covers[all_agree]

        bets_a = agree_edge > 0
        hits = bets_a == (agree_covers == 1)

        results.append({
            "scenario": "All models agree",
            "n_games": int(all_agree.sum()),
            "hit_rate": float(hits.mean()),
            "roi": float(compute_roi(hits)),
        })

    # When models disagree
    disagree = ~all_agree
    if disagree.sum() > 0:
        disagree_edge = np.mean(
            [edges[name][valid_mask][disagree] for name in edges.keys()], axis=0
        )
        disagree_covers = valid_covers[disagree]

        bets_a = disagree_edge > 0
        hits = bets_a == (disagree_covers == 1)

        results.append({
            "scenario": "Models disagree",
            "n_games": int(disagree.sum()),
            "hit_rate": float(hits.mean()),
            "roi": float(compute_roi(hits)),
        })

    return pd.DataFrame(results)


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

    # Generate predictions
    print("\nGenerating predictions...")

    splits = {"val": val_df, "test": test_df}
    results: dict = {}

    for split_name, split_df in splits.items():
        predictions = {}

        for model_name in models:
            predictions[model_name] = predict_with_model(
                model_name, models[model_name], split_df
            )

        # Create ensemble
        ensemble = predict_ensemble(models, split_df)

        results[split_name] = {
            "predictions": predictions,
            "ensemble": ensemble,
            "spreads": split_df["spread_a"].values,
            "covers": split_df["cover_a"].values,
            "margins": split_df["final_margin_a"].values,
        }

    # ========================================
    # ANALYSIS 1: Edge Bucket Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("EDGE BUCKET ANALYSIS (Validation Set)")
    print("=" * 70)

    bucket_results = analyze_by_edge_bucket(
        results["val"]["ensemble"],
        results["val"]["spreads"],
        results["val"]["covers"],
    )
    bucket_df = pd.DataFrame(bucket_results)
    print(bucket_df.to_string(index=False))

    # ========================================
    # ANALYSIS 2: Threshold Sweep on Validation
    # ========================================
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP (Validation Set)")
    print("=" * 70)

    thresholds = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    threshold_results = []

    for thresh in thresholds:
        result = evaluate_with_threshold(
            predictions=results["val"]["ensemble"],
            spreads=results["val"]["spreads"],
            covers=results["val"]["covers"],
            threshold=thresh,
        )
        threshold_results.append(result)

    threshold_df = pd.DataFrame(threshold_results)
    print(threshold_df.to_string(index=False))

    # Find optimal threshold (maximize ROI with minimum 50 bets)
    viable = threshold_df[threshold_df["n_bets"] >= 50]
    if len(viable) > 0:
        best_idx = viable["roi"].idxmax()
        optimal_threshold = threshold_df.loc[best_idx, "threshold"]
    else:
        optimal_threshold = 0.0

    print(f"\nOptimal threshold (min 50 bets): {optimal_threshold} pts")

    # ========================================
    # ANALYSIS 3: Model Agreement
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL AGREEMENT ANALYSIS (Validation Set)")
    print("=" * 70)

    agreement_df = analyze_model_agreement(
        results["val"]["predictions"],
        results["val"]["spreads"],
        results["val"]["covers"],
    )
    print(agreement_df.to_string(index=False))

    # ========================================
    # TEST SET EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE")
    print("=" * 70)

    # Test with no threshold (all bets)
    test_all = evaluate_ats(
        predictions=results["test"]["ensemble"],
        spreads=results["test"]["spreads"],
        covers=results["test"]["covers"],
        threshold=0,
    )
    print(f"\nAll bets (no threshold):")
    print(f"  Hit Rate: {test_all['hit_rate']:.3f}")
    print(f"  ROI: {test_all['roi']:.3f}")
    print(f"  N bets: {test_all['n_bets']}")

    # Test with optimal threshold from validation
    test_selective = evaluate_with_threshold(
        predictions=results["test"]["ensemble"],
        spreads=results["test"]["spreads"],
        covers=results["test"]["covers"],
        threshold=optimal_threshold,
    )
    print(f"\nSelective betting (threshold={optimal_threshold}):")
    print(f"  Hit Rate: {test_selective['hit_rate']:.3f}")
    print(f"  ROI: {test_selective['roi']:.3f}")
    print(f"  N bets: {test_selective['n_bets']}")

    # Test bucket analysis on test set
    print("\n" + "=" * 70)
    print("EDGE BUCKET ANALYSIS (Test Set)")
    print("=" * 70)

    test_bucket_results = analyze_by_edge_bucket(
        results["test"]["ensemble"],
        results["test"]["spreads"],
        results["test"]["covers"],
    )
    test_bucket_df = pd.DataFrame(test_bucket_results)
    print(test_bucket_df.to_string(index=False))

    # Model agreement on test
    print("\n" + "=" * 70)
    print("MODEL AGREEMENT ANALYSIS (Test Set)")
    print("=" * 70)

    test_agreement_df = analyze_model_agreement(
        results["test"]["predictions"],
        results["test"]["spreads"],
        results["test"]["covers"],
    )
    print(test_agreement_df.to_string(index=False))

    # ========================================
    # FINAL VERDICT
    # ========================================
    breakeven = 0.524
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    print(f"\nStrategy 1: Bet all games")
    print(f"  Hit Rate: {test_all['hit_rate']:.1%}")
    print(
        f"  Status: {'PROFITABLE' if test_all['hit_rate'] >= breakeven else 'Below breakeven'}"
    )

    print(f"\nStrategy 2: Selective betting (|edge| >= {optimal_threshold})")
    print(f"  Hit Rate: {test_selective['hit_rate']:.1%}")
    print(
        f"  N bets: {test_selective['n_bets']} ({test_selective.get('pct_games_bet', 0):.1%} of games)"
    )
    print(
        f"  Status: {'PROFITABLE' if test_selective['hit_rate'] >= breakeven else 'Below breakeven'}"
    )

    # Find ANY profitable threshold on test
    print("\n" + "=" * 70)
    print("SEARCHING FOR PROFITABLE THRESHOLDS ON TEST")
    print("=" * 70)

    for thresh in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]:
        result = evaluate_with_threshold(
            predictions=results["test"]["ensemble"],
            spreads=results["test"]["spreads"],
            covers=results["test"]["covers"],
            threshold=thresh,
        )
        status = "PROFITABLE" if result["hit_rate"] >= breakeven else ""
        print(
            f"Threshold {thresh:>4}: {result['hit_rate']:.3f} hit rate, "
            f"{result['roi']:+.3f} ROI, {result['n_bets']:>4} bets {status}"
        )

    # Fine-grained search around 4.5
    print("\n" + "=" * 70)
    print("FINE-GRAINED SEARCH (4.0 - 6.0)")
    print("=" * 70)

    for thresh in np.arange(4.0, 6.1, 0.25):
        result = evaluate_with_threshold(
            predictions=results["test"]["ensemble"],
            spreads=results["test"]["spreads"],
            covers=results["test"]["covers"],
            threshold=thresh,
        )
        status = "PROFITABLE" if result["hit_rate"] >= breakeven else ""
        print(
            f"Threshold {thresh:.2f}: {result['hit_rate']:.3f} hit rate, "
            f"{result['roi']:+.3f} ROI, {result['n_bets']:>4} bets {status}"
        )

    # Save results
    summary = {
        "all_bets_hit_rate": test_all["hit_rate"],
        "all_bets_roi": test_all["roi"],
        "all_bets_n": test_all["n_bets"],
        "selective_threshold": optimal_threshold,
        "selective_hit_rate": test_selective["hit_rate"],
        "selective_roi": test_selective["roi"],
        "selective_n": test_selective["n_bets"],
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(
        metrics_dir / "selective_betting_summary.csv", index=False
    )
    print(f"\nSaved results to {metrics_dir / 'selective_betting_summary.csv'}")


if __name__ == "__main__":
    main()

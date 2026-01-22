"""
ATS Backtesting with threshold-based bet selection.

Loads trained models and runs backtest on test data.
Tunes edge threshold on validation set.

Output:
- reports/backtests/baseline_<model>_<train>_<test>.csv
- reports/metrics/backtest_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle


def compute_roi(hits: np.ndarray, juice: float = -110) -> float:
    """
    Compute ROI assuming standard -110 juice.

    Args:
        hits: boolean array of whether each bet hit
        juice: vig (default -110)

    Returns:
        ROI as a decimal (0.05 = 5% profit)
    """
    if len(hits) == 0:
        return 0.0

    # Risk 1 unit per bet
    # Win: profit = 100/110 = 0.909
    # Lose: loss = -1
    win_payout = 100 / abs(juice)

    total_profit = hits.sum() * win_payout - (~hits).sum() * 1.0
    total_risked = len(hits)

    return total_profit / total_risked if total_risked > 0 else 0.0


def backtest_with_threshold(
    y_pred: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    edge_threshold: float
) -> dict:
    """
    Run backtest with a given edge threshold.

    Args:
        y_pred: predicted margins
        spreads: spread lines (from team A perspective)
        covers: actual cover results (1=cover, 0=no cover, 0.5=push)
        edge_threshold: minimum predicted edge to place bet

    Returns:
        dict with backtest metrics
    """
    # Compute predicted edge
    # If we predict margin > -spread, we think team A will cover
    pred_edge = y_pred - (-spreads)

    # Exclude pushes
    valid_mask = np.isin(covers, [0, 1])

    # Find bets that meet threshold
    bet_a_mask = (pred_edge >= edge_threshold) & valid_mask
    bet_b_mask = (pred_edge <= -edge_threshold) & valid_mask

    # Bet outcomes
    bet_a_hits = covers[bet_a_mask] == 1
    bet_b_hits = covers[bet_b_mask] == 0  # Betting on B means A doesn't cover

    all_hits = np.concatenate([bet_a_hits, bet_b_hits])
    total_bets = len(all_hits)

    if total_bets == 0:
        return {
            'edge_threshold': edge_threshold,
            'total_bets': 0,
            'bet_rate': 0.0,
            'hit_rate': None,
            'roi': None,
        }

    hit_rate = all_hits.mean()
    roi = compute_roi(all_hits)

    return {
        'edge_threshold': edge_threshold,
        'total_bets': total_bets,
        'bet_rate': total_bets / valid_mask.sum(),
        'hit_rate': hit_rate,
        'roi': roi,
        'bets_on_a': bet_a_mask.sum(),
        'bets_on_b': bet_b_mask.sum(),
    }


def tune_threshold(
    y_pred: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    thresholds: list[float] = None
) -> tuple[float, pd.DataFrame]:
    """
    Find optimal edge threshold on validation set.

    Returns:
        best_threshold: threshold with best ROI (with sufficient bets)
        results_df: DataFrame with all threshold results
    """
    if thresholds is None:
        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

    results = []
    for thresh in thresholds:
        result = backtest_with_threshold(y_pred, spreads, covers, thresh)
        results.append(result)

    results_df = pd.DataFrame(results)

    # Find best threshold: highest ROI with at least 100 bets
    valid_results = results_df[results_df['total_bets'] >= 100]
    if len(valid_results) > 0:
        best_idx = valid_results['roi'].idxmax()
        best_threshold = valid_results.loc[best_idx, 'edge_threshold']
    else:
        # Fall back to threshold=0 if not enough bets at higher thresholds
        best_threshold = 0.0

    return best_threshold, results_df


def backtest_by_edge_bucket(
    y_pred: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    buckets: list[tuple] = None
) -> pd.DataFrame:
    """
    Analyze hit rate by predicted edge bucket.

    This is a calibration check to see if larger predicted edges
    correspond to higher hit rates.
    """
    if buckets is None:
        buckets = [
            (-float('inf'), -5),
            (-5, -3),
            (-3, -1),
            (-1, 1),
            (1, 3),
            (3, 5),
            (5, float('inf')),
        ]

    pred_edge = y_pred - (-spreads)
    valid_mask = np.isin(covers, [0, 1])

    results = []
    for low, high in buckets:
        mask = (pred_edge >= low) & (pred_edge < high) & valid_mask

        if mask.sum() == 0:
            continue

        # For positive edge buckets, we'd bet on A
        # For negative edge buckets, we'd bet on B
        if low >= 0:
            hits = covers[mask] == 1  # A covers
        else:
            hits = covers[mask] == 0  # A doesn't cover (B bet wins)

        results.append({
            'edge_bucket': f'[{low}, {high})',
            'n_games': mask.sum(),
            'hit_rate': hits.mean(),
            'roi': compute_roi(hits),
        })

    return pd.DataFrame(results)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'reports' / 'models'
    backtests_dir = project_root / 'reports' / 'backtests'
    metrics_dir = project_root / 'reports' / 'metrics'

    backtests_dir.mkdir(parents=True, exist_ok=True)

    # Import train_ats_baseline to use its feature preparation
    from src.cbb.train_ats_baseline import compute_rolling_team_stats, prepare_features

    # Load games
    games_df = pd.read_parquet(processed_dir / 'games_base.parquet')
    games_df = games_df[games_df['spread_a'].notna()].copy()
    print(f"Loaded {len(games_df):,} games with spread data")

    # Compute rolling team stats
    print("Computing rolling team stats...")
    team_stats = compute_rolling_team_stats(games_df, window=10)

    # Prepare features
    print("Preparing features...")
    features_df = prepare_features(games_df, team_stats)
    features_df = features_df.dropna()

    # Define feature columns
    feature_cols = [
        'team_a_rolling_margin', 'team_a_rolling_ppg', 'team_a_rolling_papg',
        'team_b_rolling_margin', 'team_b_rolling_ppg', 'team_b_rolling_papg',
        'margin_diff', 'ppg_diff', 'papg_diff',
        'is_home_a', 'is_neutral',
    ]

    # Split data
    val_df = features_df[features_df['season'] == '2024-25']
    test_df = features_df[features_df['season'] == '2025-26']

    print(f"Val: {len(val_df):,} games")
    print(f"Test: {len(test_df):,} games")

    # Load models
    with open(models_dir / 'ridge_baseline.pkl', 'rb') as f:
        ridge_data = pickle.load(f)
    ridge = ridge_data['model']
    scaler = ridge_data['scaler']

    with open(models_dir / 'gbm_baseline.pkl', 'rb') as f:
        gbm_data = pickle.load(f)
    gbm = gbm_data['model']

    all_results = []

    for model_name, model, use_scaler in [('Ridge', ridge, True), ('GBM', gbm, False)]:
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {model_name}")
        print('='*60)

        # Get predictions for val and test
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values

        if use_scaler:
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        spreads_val = val_df['spread_a'].values
        spreads_test = test_df['spread_a'].values
        covers_val = val_df['cover_a'].values
        covers_test = test_df['cover_a'].values

        # Tune threshold on validation
        print("\nTuning threshold on validation set...")
        best_threshold, val_results = tune_threshold(y_pred_val, spreads_val, covers_val)
        print(val_results.to_string(index=False))
        print(f"\nBest threshold: {best_threshold}")

        # Backtest on test with best threshold
        print(f"\nTest set backtest (threshold={best_threshold})...")
        test_result = backtest_with_threshold(y_pred_test, spreads_test, covers_test, best_threshold)
        print(f"  Total bets: {test_result['total_bets']:,}")
        print(f"  Bet rate: {test_result['bet_rate']:.1%}")
        print(f"  Hit rate: {test_result['hit_rate']:.1%}" if test_result['hit_rate'] else "  Hit rate: N/A")
        print(f"  ROI: {test_result['roi']:.1%}" if test_result['roi'] else "  ROI: N/A")

        # Edge bucket analysis
        print("\nEdge bucket calibration (test set):")
        bucket_results = backtest_by_edge_bucket(y_pred_test, spreads_test, covers_test)
        print(bucket_results.to_string(index=False))

        # Save detailed results
        test_details = test_df.copy()
        test_details['y_pred'] = model.predict(X_test if not use_scaler else scaler.transform(test_df[feature_cols].values))
        test_details['pred_edge'] = test_details['y_pred'] - (-test_details['spread_a'])
        test_details.to_csv(backtests_dir / f'baseline_{model_name.lower()}_2022_2026.csv', index=False)

        all_results.append({
            'model': model_name,
            'train_seasons': '2022-23, 2023-24',
            'val_season': '2024-25',
            'test_season': '2025-26',
            'best_threshold': best_threshold,
            'test_bets': test_result['total_bets'],
            'test_bet_rate': test_result['bet_rate'],
            'test_hit_rate': test_result['hit_rate'],
            'test_roi': test_result['roi'],
        })

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(metrics_dir / 'backtest_summary.csv', index=False)
    print(f"\n✓ Saved backtest results to {backtests_dir}")
    print(f"✓ Saved summary to {metrics_dir / 'backtest_summary.csv'}")

    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()

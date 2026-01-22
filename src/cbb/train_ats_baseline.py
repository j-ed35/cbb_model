"""
Baseline ATS model training.

Trains Ridge and GradientBoosting regressors to predict game margin.
Uses time-based train/val/test split.

Modes:
- Default: Uses rolling team stats, time-based season splits
- --kenpom-cv: Uses KenPom features with cross-validation (for small datasets)

Output:
- reports/models/ridge_baseline.pkl
- reports/models/gbm_baseline.pkl
- reports/metrics/baseline_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import argparse


def compute_rolling_team_stats(games_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute rolling team statistics for use as features.

    This creates time-safe features by only using data from games
    played BEFORE the current game.
    """
    # Sort by date
    games_df = games_df.sort_values('date').reset_index(drop=True)

    # Create team-level stats
    team_games = []

    # Process team_a's perspective
    for idx, row in games_df.iterrows():
        team_games.append({
            'date': row['date'],
            'team': row['team_a'],
            'margin': row['final_margin_a'],
            'points_for': row['points_a'],
            'points_against': row['points_b'],
            'is_home': 1 if not row['is_neutral'] else 0,
        })
        # Also add team_b's perspective
        team_games.append({
            'date': row['date'],
            'team': row['team_b'],
            'margin': -row['final_margin_a'],
            'points_for': row['points_b'],
            'points_against': row['points_a'],
            'is_home': 0 if not row['is_neutral'] else 0,
        })

    team_df = pd.DataFrame(team_games)
    team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

    # Compute rolling stats per team
    # Use shift(1) to ensure we only use past data
    stats = team_df.groupby('team').apply(
        lambda x: x.assign(
            rolling_margin=x['margin'].shift(1).rolling(window, min_periods=3).mean(),
            rolling_ppg=x['points_for'].shift(1).rolling(window, min_periods=3).mean(),
            rolling_papg=x['points_against'].shift(1).rolling(window, min_periods=3).mean(),
            games_played=x.groupby('team').cumcount(),
        )
    ).reset_index(drop=True)

    return stats


def prepare_features(
    games_df: pd.DataFrame,
    team_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare feature matrix for modeling using merge.
    """
    # Deduplicate team_stats by taking first entry per (team, date)
    # This handles cases where a team plays multiple games on the same date
    team_stats_dedup = team_stats.drop_duplicates(subset=['team', 'date'], keep='first')

    # Prepare team A stats
    team_a_stats = team_stats_dedup[['team', 'date', 'rolling_margin', 'rolling_ppg', 'rolling_papg']].copy()
    team_a_stats.columns = ['team_a', 'date', 'team_a_rolling_margin', 'team_a_rolling_ppg', 'team_a_rolling_papg']

    # Prepare team B stats
    team_b_stats = team_stats_dedup[['team', 'date', 'rolling_margin', 'rolling_ppg', 'rolling_papg']].copy()
    team_b_stats.columns = ['team_b', 'date', 'team_b_rolling_margin', 'team_b_rolling_ppg', 'team_b_rolling_papg']

    # Start with games
    features_df = games_df[['game_key', 'date', 'season', 'team_a', 'team_b',
                            'final_margin_a', 'spread_a', 'cover_a', 'is_neutral']].copy()

    # Merge team A stats
    features_df = features_df.merge(team_a_stats, on=['team_a', 'date'], how='left')

    # Merge team B stats
    features_df = features_df.merge(team_b_stats, on=['team_b', 'date'], how='left')

    # Compute derived features
    features_df['margin_diff'] = features_df['team_a_rolling_margin'] - features_df['team_b_rolling_margin']
    features_df['ppg_diff'] = features_df['team_a_rolling_ppg'] - features_df['team_b_rolling_ppg']
    features_df['papg_diff'] = features_df['team_a_rolling_papg'] - features_df['team_b_rolling_papg']

    # Context features
    features_df['is_home_a'] = (~features_df['is_neutral']).astype(int)
    features_df['is_neutral'] = features_df['is_neutral'].astype(int)

    return features_df


def train_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = 'final_margin_a'
) -> tuple[Ridge, GradientBoostingRegressor, StandardScaler]:
    """
    Train Ridge and GradientBoosting models.
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    # Scale features for Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    print(f"Ridge training MAE: {mean_absolute_error(y_train, ridge.predict(X_train_scaled)):.2f}")

    # Train GradientBoosting (sklearn version, no libomp needed)
    gbm_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    gbm_model.fit(X_train, y_train)
    print(f"GBM training MAE: {mean_absolute_error(y_train, gbm_model.predict(X_train)):.2f}")

    return ridge, gbm_model, scaler


def evaluate_model(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler = None,
    model_name: str = 'Model'
) -> dict:
    """
    Evaluate model performance on a dataset.
    """
    X = df[feature_cols].values
    y_true = df['final_margin_a'].values
    spreads = df['spread_a'].values
    covers = df['cover_a'].values

    if scaler is not None:
        X = scaler.transform(X)

    y_pred = model.predict(X)

    # Regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Compute predicted edge
    pred_edge = y_pred - (-spreads)  # spread_a is from team A perspective

    # ATS metrics (excluding pushes)
    valid_mask = np.isin(covers, [0, 1])
    valid_pred_edge = pred_edge[valid_mask]
    valid_covers = covers[valid_mask]

    # Simple: bet on team A if pred_edge > 0
    simple_bets = valid_pred_edge > 0
    simple_hits = (simple_bets == (valid_covers == 1))
    simple_hit_rate = simple_hits.mean()

    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'n_games': len(df),
        'n_valid_ats': valid_mask.sum(),
        'simple_hit_rate': simple_hit_rate,
    }


def run_kenpom_cv_mode(project_root: Path):
    """
    Run cross-validation on KenPom-matched games only.

    Use this mode when you only have limited KenPom data (e.g., daily snapshots
    for a short time window). This validates that KenPom features are predictive
    before investing time in scraping historical data.
    """
    features_dir = project_root / 'data' / 'features'
    metrics_dir = project_root / 'reports' / 'metrics'

    # Load features file
    features_df = pd.read_parquet(features_dir / 'games_features.parquet')

    # Filter to games with KenPom match AND spread data
    kp_df = features_df[
        (features_df['kp_matched'] == True) &
        (features_df['spread_a'].notna())
    ].copy()

    print(f"Games with KenPom + spread: {len(kp_df)}")
    if len(kp_df) < 50:
        print("ERROR: Too few games for meaningful analysis. Need at least 50.")
        return

    print(f"Date range: {kp_df['date'].min()} to {kp_df['date'].max()}")

    # Define KenPom feature columns
    kp_feature_cols = [
        'kp_adj_em_a', 'kp_adj_em_b', 'kp_adj_em_diff',
        'kp_adj_o_a', 'kp_adj_o_b',
        'kp_adj_d_a', 'kp_adj_d_b',
        'kp_tempo_a', 'kp_tempo_b', 'kp_tempo_avg',
        'is_home_a', 'is_neutral',
    ]

    # Check which columns exist
    available_cols = [c for c in kp_feature_cols if c in kp_df.columns]
    missing_cols = [c for c in kp_feature_cols if c not in kp_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")

    print(f"Using features: {available_cols}")

    # Drop rows with missing features
    kp_df = kp_df.dropna(subset=available_cols + ['final_margin_a'])
    print(f"After dropping NaN: {len(kp_df)} games")

    if len(kp_df) < 50:
        print("ERROR: Too few complete rows. Check KenPom feature extraction.")
        return

    X = kp_df[available_cols].values
    y = kp_df['final_margin_a'].values
    spreads = kp_df['spread_a'].values
    covers = kp_df['cover_a'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("KENPOM CROSS-VALIDATION RESULTS")
    print("=" * 60)

    # Ridge CV
    ridge = Ridge(alpha=1.0)
    ridge_preds = cross_val_predict(ridge, X_scaled, y, cv=kfold)
    ridge_mae = mean_absolute_error(y, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y, ridge_preds))

    # GBM CV
    gbm = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gbm_preds = cross_val_predict(gbm, X, y, cv=kfold)
    gbm_mae = mean_absolute_error(y, gbm_preds)
    gbm_rmse = np.sqrt(mean_squared_error(y, gbm_preds))

    # ATS evaluation (excluding pushes)
    valid_mask = np.isin(covers, [0, 1])

    def ats_hit_rate(preds, spreads, covers, mask):
        pred_edge = preds[mask] - (-spreads[mask])
        bets_a = pred_edge > 0
        hits = (bets_a == (covers[mask] == 1))
        return hits.mean()

    ridge_ats = ats_hit_rate(ridge_preds, spreads, covers, valid_mask)
    gbm_ats = ats_hit_rate(gbm_preds, spreads, covers, valid_mask)

    print(f"\nRidge (5-fold CV):")
    print(f"  MAE:  {ridge_mae:.2f}")
    print(f"  RMSE: {ridge_rmse:.2f}")
    print(f"  ATS Hit Rate: {ridge_ats:.1%}")

    print(f"\nGBM (5-fold CV):")
    print(f"  MAE:  {gbm_mae:.2f}")
    print(f"  RMSE: {gbm_rmse:.2f}")
    print(f"  ATS Hit Rate: {gbm_ats:.1%}")

    # Correlation check
    em_diff = kp_df['kp_adj_em_diff'].values if 'kp_adj_em_diff' in kp_df.columns else None
    if em_diff is not None:
        corr_margin = np.corrcoef(em_diff, y)[0, 1]
        corr_spread = np.corrcoef(em_diff, -spreads)[0, 1]
        print(f"\nKenPom AdjEM Diff correlations:")
        print(f"  vs Actual Margin: {corr_margin:.3f}")
        print(f"  vs Spread:        {corr_spread:.3f}")
        print(f"  (Higher margin corr than spread corr suggests potential edge)")

    # Save results
    results = {
        'mode': 'kenpom_cv',
        'n_games': len(kp_df),
        'ridge_mae': ridge_mae,
        'ridge_rmse': ridge_rmse,
        'ridge_ats': ridge_ats,
        'gbm_mae': gbm_mae,
        'gbm_rmse': gbm_rmse,
        'gbm_ats': gbm_ats,
    }
    pd.DataFrame([results]).to_csv(metrics_dir / 'kenpom_cv_summary.csv', index=False)
    print(f"\n✓ Saved results to {metrics_dir / 'kenpom_cv_summary.csv'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ATS baseline models')
    parser.add_argument('--kenpom-cv', action='store_true',
                        help='Run cross-validation on KenPom-matched games only')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    if args.kenpom_cv:
        run_kenpom_cv_mode(project_root)
        return

    processed_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'reports' / 'models'
    metrics_dir = project_root / 'reports' / 'metrics'

    models_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Created {len(features_df):,} feature rows")

    # Define feature columns
    feature_cols = [
        'team_a_rolling_margin', 'team_a_rolling_ppg', 'team_a_rolling_papg',
        'team_b_rolling_margin', 'team_b_rolling_ppg', 'team_b_rolling_papg',
        'margin_diff', 'ppg_diff', 'papg_diff',
        'is_home_a', 'is_neutral',
    ]

    # Time-based split
    # Train: 2022-23, 2023-24
    # Val: 2024-25
    # Test: 2025-26
    train_df = features_df[features_df['season'].isin(['2022-23', '2023-24'])]
    val_df = features_df[features_df['season'] == '2024-25']
    test_df = features_df[features_df['season'] == '2025-26']

    print(f"\nTrain: {len(train_df):,} games ({train_df['season'].unique()})")
    print(f"Val: {len(val_df):,} games ({val_df['season'].unique()})")
    print(f"Test: {len(test_df):,} games ({test_df['season'].unique()})")

    # Train models
    print("\nTraining models...")
    ridge, gbm_model, scaler = train_models(train_df, feature_cols)

    # Evaluate on all sets
    results = []

    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        ridge_result = evaluate_model(ridge, df, feature_cols, scaler, f'Ridge_{name}')
        gbm_result = evaluate_model(gbm_model, df, feature_cols, None, f'GBM_{name}')
        results.extend([ridge_result, gbm_result])

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save models
    with open(models_dir / 'ridge_baseline.pkl', 'wb') as f:
        pickle.dump({'model': ridge, 'scaler': scaler, 'feature_cols': feature_cols}, f)

    with open(models_dir / 'gbm_baseline.pkl', 'wb') as f:
        pickle.dump({'model': gbm_model, 'feature_cols': feature_cols}, f)

    # Save results
    results_df.to_csv(metrics_dir / 'baseline_summary.csv', index=False)
    print(f"\n✓ Saved models to {models_dir}")
    print(f"✓ Saved results to {metrics_dir / 'baseline_summary.csv'}")

    return ridge, gbm_model, scaler, results_df


if __name__ == '__main__':
    main()

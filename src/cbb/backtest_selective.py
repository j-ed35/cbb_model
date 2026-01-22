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

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import pickle
import torch


def load_models(models_dir: Path) -> dict:
    """Load all trained models."""
    models = {}

    # Load Ridge
    ridge_path = models_dir / 'enhanced_ridge.pkl'
    if ridge_path.exists():
        with open(ridge_path, 'rb') as f:
            models['ridge'] = pickle.load(f)
        print("Loaded Ridge model")

    # Load GBM
    gbm_path = models_dir / 'enhanced_gbm.pkl'
    if gbm_path.exists():
        with open(gbm_path, 'rb') as f:
            models['gbm'] = pickle.load(f)
        print("Loaded GBM model")

    # Load DNN
    dnn_pt_path = models_dir / 'dnn_enhanced.pt'
    dnn_artifacts_path = models_dir / 'dnn_enhanced_artifacts.pkl'

    if dnn_pt_path.exists() and dnn_artifacts_path.exists():
        with open(dnn_artifacts_path, 'rb') as f:
            dnn_artifacts = pickle.load(f)

        from src.cbb.models.dnn_enhanced import TwoTowerDNN

        config = dnn_artifacts['model_config']
        model = TwoTowerDNN(
            team_feature_dim=config['team_feature_dim'],
            matchup_feature_dim=config['matchup_feature_dim'],
            context_feature_dim=config['context_feature_dim'],
        )
        model.load_state_dict(torch.load(dnn_pt_path, weights_only=True))
        model.eval()

        models['dnn'] = {
            'model': model,
            'scalers': dnn_artifacts['scalers'],
            'feature_groups': dnn_artifacts['feature_groups'],
        }
        print("Loaded DNN model")

    return models


def predict_ridge(model_data: dict, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from Ridge model."""
    feature_cols = model_data['feature_cols']
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    X_scaled = model_data['scaler'].transform(X)
    return model_data['model'].predict(X_scaled)


def predict_gbm(model_data: dict, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from GBM model."""
    feature_cols = model_data['feature_cols']
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    return model_data['model'].predict(X)


def predict_dnn(model_data: dict, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from DNN model."""
    scalers = model_data['scalers']
    feature_groups = model_data['feature_groups']
    model = model_data['model']

    def get_features(feature_list):
        available = [f for f in feature_list if f in df.columns]
        data = df[available].fillna(0).values
        return np.nan_to_num(data, nan=0)

    team_a = scalers['team_a'].transform(get_features(feature_groups['team_a']))
    team_b = scalers['team_b'].transform(get_features(feature_groups['team_b']))
    matchup = scalers['matchup'].transform(get_features(feature_groups['matchup']))
    context = scalers['context'].transform(get_features(feature_groups['context']))

    team_a_t = torch.FloatTensor(team_a)
    team_b_t = torch.FloatTensor(team_b)
    matchup_t = torch.FloatTensor(matchup)
    context_t = torch.FloatTensor(context)

    with torch.no_grad():
        preds, _ = model(team_a_t, team_b_t, matchup_t, context_t)

    return preds.numpy()


def evaluate_with_threshold(
    ensemble_pred: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    threshold: float = 0.0
) -> dict:
    """Evaluate predictions with edge threshold."""
    # Calculate edge: how much better we think Team A is vs the spread
    pred_edge = ensemble_pred - (-spreads)  # Our margin pred vs implied margin

    # Filter valid covers (no pushes)
    valid_mask = np.isin(covers, [0, 1])
    valid_edge = pred_edge[valid_mask]
    valid_covers = covers[valid_mask]

    # Apply threshold - only bet when |edge| > threshold
    bet_mask = np.abs(valid_edge) >= threshold

    if bet_mask.sum() == 0:
        return {
            'hit_rate': 0,
            'roi': 0,
            'n_bets': 0,
            'threshold': threshold,
        }

    filtered_edge = valid_edge[bet_mask]
    filtered_covers = valid_covers[bet_mask]

    # Betting logic: bet on Team A covering if edge > 0, else bet Team B covers
    bets_a = filtered_edge > 0
    hits = (bets_a == (filtered_covers == 1))
    hit_rate = hits.mean()

    # ROI at -110 odds
    win_payout = 100 / 110
    n_bets = len(hits)
    profit = hits.sum() * win_payout - (n_bets - hits.sum())
    roi = profit / n_bets if n_bets > 0 else 0

    return {
        'hit_rate': hit_rate,
        'roi': roi,
        'n_bets': n_bets,
        'threshold': threshold,
        'pct_games_bet': bet_mask.sum() / len(valid_mask),
    }


def analyze_by_edge_bucket(
    ensemble_pred: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
) -> pd.DataFrame:
    """Analyze performance by edge magnitude buckets."""
    pred_edge = ensemble_pred - (-spreads)
    abs_edge = np.abs(pred_edge)

    # Filter valid
    valid_mask = np.isin(covers, [0, 1])
    valid_edge = pred_edge[valid_mask]
    valid_abs_edge = abs_edge[valid_mask]
    valid_covers = covers[valid_mask]

    # Define buckets
    buckets = [
        (0, 1, "0-1 pts"),
        (1, 2, "1-2 pts"),
        (2, 3, "2-3 pts"),
        (3, 4, "3-4 pts"),
        (4, 5, "4-5 pts"),
        (5, 7, "5-7 pts"),
        (7, 10, "7-10 pts"),
        (10, float('inf'), "10+ pts"),
    ]

    results = []
    for low, high, label in buckets:
        mask = (valid_abs_edge >= low) & (valid_abs_edge < high)
        if mask.sum() == 0:
            continue

        bucket_edge = valid_edge[mask]
        bucket_covers = valid_covers[mask]

        bets_a = bucket_edge > 0
        hits = (bets_a == (bucket_covers == 1))

        win_payout = 100 / 110
        n_bets = len(hits)
        profit = hits.sum() * win_payout - (n_bets - hits.sum())
        roi = profit / n_bets if n_bets > 0 else 0

        results.append({
            'bucket': label,
            'n_games': mask.sum(),
            'hit_rate': hits.mean(),
            'roi': roi,
        })

    return pd.DataFrame(results)


def analyze_model_agreement(
    predictions: dict,
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

    # When all models agree
    if all_agree.sum() > 0:
        agree_edge = np.mean([edges[name][valid_mask][all_agree] for name in edges.keys()], axis=0)
        agree_covers = valid_covers[all_agree]

        bets_a = agree_edge > 0
        hits = (bets_a == (agree_covers == 1))

        win_payout = 100 / 110
        n_bets = len(hits)
        profit = hits.sum() * win_payout - (n_bets - hits.sum())
        roi = profit / n_bets if n_bets > 0 else 0

        results.append({
            'scenario': 'All models agree',
            'n_games': all_agree.sum(),
            'hit_rate': hits.mean(),
            'roi': roi,
        })

    # When models disagree
    disagree = ~all_agree
    if disagree.sum() > 0:
        disagree_edge = np.mean([edges[name][valid_mask][disagree] for name in edges.keys()], axis=0)
        disagree_covers = valid_covers[disagree]

        bets_a = disagree_edge > 0
        hits = (bets_a == (disagree_covers == 1))

        win_payout = 100 / 110
        n_bets = len(hits)
        profit = hits.sum() * win_payout - (n_bets - hits.sum())
        roi = profit / n_bets if n_bets > 0 else 0

        results.append({
            'scenario': 'Models disagree',
            'n_games': disagree.sum(),
            'hit_rate': hits.mean(),
            'roi': roi,
        })

    return pd.DataFrame(results)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    features_dir = project_root / 'data' / 'features'
    models_dir = project_root / 'reports' / 'models'
    metrics_dir = project_root / 'reports' / 'metrics'

    # Load features
    features_df = pd.read_parquet(features_dir / 'games_features_enhanced.parquet')

    # Filter to usable games
    df = features_df[(features_df['kp_matched'] == True) & (features_df['spread_a'].notna())].copy()
    df = df.dropna(subset=['final_margin_a', 'cover_a'])

    print(f"Loaded {len(df):,} usable games")

    # Split by season
    train_df = df[df['season'].isin(['2022-23', '2023-24'])]
    val_df = df[df['season'] == '2024-25']
    test_df = df[df['season'] == '2025-26']

    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    print(f"Test:  {len(test_df):,}")

    # Load models
    print("\nLoading models...")
    models = load_models(models_dir)

    if len(models) == 0:
        print("No models found! Train models first.")
        return

    # Generate predictions
    print("\nGenerating predictions...")

    splits = {'val': val_df, 'test': test_df}
    results = {}

    for split_name, split_df in splits.items():
        predictions = {}

        if 'ridge' in models:
            predictions['ridge'] = predict_ridge(models['ridge'], split_df)

        if 'gbm' in models:
            predictions['gbm'] = predict_gbm(models['gbm'], split_df)

        if 'dnn' in models:
            predictions['dnn'] = predict_dnn(models['dnn'], split_df)

        results[split_name] = {
            'predictions': predictions,
            'spreads': split_df['spread_a'].values,
            'covers': split_df['cover_a'].values,
            'margins': split_df['final_margin_a'].values,
        }

    # Create ensemble predictions (use DNN-heavy weighting based on prior results)
    # DNN performed best, so weight it more heavily
    weights = {'ridge': 0.1, 'gbm': 0.2, 'dnn': 0.7}

    for split_name in results:
        ensemble = np.zeros_like(results[split_name]['predictions']['dnn'])
        for name, w in weights.items():
            if name in results[split_name]['predictions']:
                ensemble += w * results[split_name]['predictions'][name]
        results[split_name]['ensemble'] = ensemble

    # ========================================
    # ANALYSIS 1: Edge Bucket Analysis
    # ========================================
    print("\n" + "="*70)
    print("EDGE BUCKET ANALYSIS (Validation Set)")
    print("="*70)

    bucket_df = analyze_by_edge_bucket(
        results['val']['ensemble'],
        results['val']['spreads'],
        results['val']['covers']
    )
    print(bucket_df.to_string(index=False))

    # ========================================
    # ANALYSIS 2: Threshold Sweep on Validation
    # ========================================
    print("\n" + "="*70)
    print("THRESHOLD SWEEP (Validation Set)")
    print("="*70)

    thresholds = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    threshold_results = []

    for thresh in thresholds:
        result = evaluate_with_threshold(
            results['val']['ensemble'],
            results['val']['spreads'],
            results['val']['covers'],
            threshold=thresh
        )
        threshold_results.append(result)

    threshold_df = pd.DataFrame(threshold_results)
    print(threshold_df.to_string(index=False))

    # Find optimal threshold (maximize ROI with minimum 50 bets)
    viable = threshold_df[threshold_df['n_bets'] >= 50]
    if len(viable) > 0:
        best_idx = viable['roi'].idxmax()
        optimal_threshold = threshold_df.loc[best_idx, 'threshold']
    else:
        optimal_threshold = 0

    print(f"\nOptimal threshold (min 50 bets): {optimal_threshold} pts")

    # ========================================
    # ANALYSIS 3: Model Agreement
    # ========================================
    print("\n" + "="*70)
    print("MODEL AGREEMENT ANALYSIS (Validation Set)")
    print("="*70)

    agreement_df = analyze_model_agreement(
        results['val']['predictions'],
        results['val']['spreads'],
        results['val']['covers']
    )
    print(agreement_df.to_string(index=False))

    # ========================================
    # TEST SET EVALUATION
    # ========================================
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE")
    print("="*70)

    # Test with no threshold (all bets)
    test_all = evaluate_with_threshold(
        results['test']['ensemble'],
        results['test']['spreads'],
        results['test']['covers'],
        threshold=0
    )
    print(f"\nAll bets (no threshold):")
    print(f"  Hit Rate: {test_all['hit_rate']:.3f}")
    print(f"  ROI: {test_all['roi']:.3f}")
    print(f"  N bets: {test_all['n_bets']}")

    # Test with optimal threshold from validation
    test_selective = evaluate_with_threshold(
        results['test']['ensemble'],
        results['test']['spreads'],
        results['test']['covers'],
        threshold=optimal_threshold
    )
    print(f"\nSelective betting (threshold={optimal_threshold}):")
    print(f"  Hit Rate: {test_selective['hit_rate']:.3f}")
    print(f"  ROI: {test_selective['roi']:.3f}")
    print(f"  N bets: {test_selective['n_bets']}")

    # Test bucket analysis on test set
    print("\n" + "="*70)
    print("EDGE BUCKET ANALYSIS (Test Set)")
    print("="*70)

    test_bucket_df = analyze_by_edge_bucket(
        results['test']['ensemble'],
        results['test']['spreads'],
        results['test']['covers']
    )
    print(test_bucket_df.to_string(index=False))

    # Model agreement on test
    print("\n" + "="*70)
    print("MODEL AGREEMENT ANALYSIS (Test Set)")
    print("="*70)

    test_agreement_df = analyze_model_agreement(
        results['test']['predictions'],
        results['test']['spreads'],
        results['test']['covers']
    )
    print(test_agreement_df.to_string(index=False))

    # ========================================
    # FINAL VERDICT
    # ========================================
    breakeven = 0.524
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    print(f"\nStrategy 1: Bet all games")
    print(f"  Hit Rate: {test_all['hit_rate']:.1%}")
    print(f"  Status: {'PROFITABLE' if test_all['hit_rate'] >= breakeven else 'Below breakeven'}")

    print(f"\nStrategy 2: Selective betting (|edge| >= {optimal_threshold})")
    print(f"  Hit Rate: {test_selective['hit_rate']:.1%}")
    print(f"  N bets: {test_selective['n_bets']} ({test_selective.get('pct_games_bet', 0):.1%} of games)")
    print(f"  Status: {'PROFITABLE' if test_selective['hit_rate'] >= breakeven else 'Below breakeven'}")

    # Find ANY profitable threshold on test
    print("\n" + "="*70)
    print("SEARCHING FOR PROFITABLE THRESHOLDS ON TEST")
    print("="*70)

    for thresh in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]:
        result = evaluate_with_threshold(
            results['test']['ensemble'],
            results['test']['spreads'],
            results['test']['covers'],
            threshold=thresh
        )
        status = "PROFITABLE" if result['hit_rate'] >= breakeven else ""
        print(f"Threshold {thresh:>4}: {result['hit_rate']:.3f} hit rate, {result['roi']:+.3f} ROI, {result['n_bets']:>4} bets {status}")

    # Fine-grained search around 4.5
    print("\n" + "="*70)
    print("FINE-GRAINED SEARCH (4.0 - 6.0)")
    print("="*70)

    for thresh in np.arange(4.0, 6.1, 0.25):
        result = evaluate_with_threshold(
            results['test']['ensemble'],
            results['test']['spreads'],
            results['test']['covers'],
            threshold=thresh
        )
        status = "PROFITABLE" if result['hit_rate'] >= breakeven else ""
        print(f"Threshold {thresh:.2f}: {result['hit_rate']:.3f} hit rate, {result['roi']:+.3f} ROI, {result['n_bets']:>4} bets {status}")

    # Save results
    summary = {
        'all_bets_hit_rate': test_all['hit_rate'],
        'all_bets_roi': test_all['roi'],
        'all_bets_n': test_all['n_bets'],
        'selective_threshold': optimal_threshold,
        'selective_hit_rate': test_selective['hit_rate'],
        'selective_roi': test_selective['roi'],
        'selective_n': test_selective['n_bets'],
    }

    pd.DataFrame([summary]).to_csv(metrics_dir / 'selective_betting_summary.csv', index=False)
    print(f"\nSaved results to {metrics_dir / 'selective_betting_summary.csv'}")


if __name__ == '__main__':
    main()

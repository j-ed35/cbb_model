"""
Final ensemble model combining Ridge, GBM, and DNN predictions.

This script:
1. Loads the three trained models
2. Creates weighted ensemble predictions
3. Tunes weights on validation set
4. Reports final test performance

Target: >52.4% hit rate (breakeven for -110 odds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
import torch
from scipy.optimize import minimize


def load_models(models_dir: Path) -> dict:
    """Load all trained models."""
    models = {}

    # Load Ridge
    ridge_path = models_dir / 'enhanced_ridge.pkl'
    if ridge_path.exists():
        with open(ridge_path, 'rb') as f:
            models['ridge'] = pickle.load(f)
        print("✓ Loaded Ridge model")

    # Load GBM
    gbm_path = models_dir / 'enhanced_gbm.pkl'
    if gbm_path.exists():
        with open(gbm_path, 'rb') as f:
            models['gbm'] = pickle.load(f)
        print("✓ Loaded GBM model")

    # Load DNN
    dnn_pt_path = models_dir / 'dnn_enhanced.pt'
    dnn_artifacts_path = models_dir / 'dnn_enhanced_artifacts.pkl'

    if dnn_pt_path.exists() and dnn_artifacts_path.exists():
        with open(dnn_artifacts_path, 'rb') as f:
            dnn_artifacts = pickle.load(f)

        # Import DNN model class
        from src.cbb.models.dnn_enhanced import TwoTowerDNN, TEAM_A_FEATURES, TEAM_B_FEATURES, MATCHUP_FEATURES, CONTEXT_FEATURES

        config = dnn_artifacts['model_config']
        model = TwoTowerDNN(
            team_feature_dim=config['team_feature_dim'],
            matchup_feature_dim=config['matchup_feature_dim'],
            context_feature_dim=config['context_feature_dim'],
        )
        model.load_state_dict(torch.load(dnn_pt_path))
        model.eval()

        models['dnn'] = {
            'model': model,
            'scalers': dnn_artifacts['scalers'],
            'feature_groups': dnn_artifacts['feature_groups'],
        }
        print("✓ Loaded DNN model")

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

    # Prepare features
    def get_features(feature_list):
        available = [f for f in feature_list if f in df.columns]
        data = df[available].fillna(0).values
        return np.nan_to_num(data, nan=0)

    team_a = scalers['team_a'].transform(get_features(feature_groups['team_a']))
    team_b = scalers['team_b'].transform(get_features(feature_groups['team_b']))
    matchup = scalers['matchup'].transform(get_features(feature_groups['matchup']))
    context = scalers['context'].transform(get_features(feature_groups['context']))

    # Convert to tensors
    team_a_t = torch.FloatTensor(team_a)
    team_b_t = torch.FloatTensor(team_b)
    matchup_t = torch.FloatTensor(matchup)
    context_t = torch.FloatTensor(context)

    # Predict
    with torch.no_grad():
        preds, _ = model(team_a_t, team_b_t, matchup_t, context_t)

    return preds.numpy()


def evaluate_ensemble(
    predictions: dict,
    weights: dict,
    spreads: np.ndarray,
    covers: np.ndarray
) -> dict:
    """Evaluate weighted ensemble."""
    # Combine predictions
    ensemble_pred = np.zeros_like(list(predictions.values())[0])
    for name, weight in weights.items():
        if name in predictions:
            ensemble_pred += weight * predictions[name]

    # Calculate edge
    pred_edge = ensemble_pred - (-spreads)

    # Filter valid covers (no pushes)
    valid_mask = np.isin(covers, [0, 1])
    valid_edge = pred_edge[valid_mask]
    valid_covers = covers[valid_mask]

    # Simple betting: bet A if edge > 0
    bets_a = valid_edge > 0
    hits = (bets_a == (valid_covers == 1))
    hit_rate = hits.mean()

    # ROI
    win_payout = 100 / 110
    n_bets = len(hits)
    profit = hits.sum() * win_payout - (n_bets - hits.sum())
    roi = profit / n_bets if n_bets > 0 else 0

    # MAE
    mae = mean_absolute_error(covers[valid_mask], ensemble_pred[valid_mask])

    return {
        'hit_rate': hit_rate,
        'roi': roi,
        'n_bets': n_bets,
        'mae': mae,
        'predictions': ensemble_pred,
    }


def tune_weights(
    predictions: dict,
    spreads: np.ndarray,
    covers: np.ndarray
) -> dict:
    """Tune ensemble weights to maximize hit rate."""
    model_names = list(predictions.keys())
    n_models = len(model_names)

    def objective(w):
        # Ensure weights sum to 1
        w = w / w.sum()
        weights = {name: w[i] for i, name in enumerate(model_names)}
        result = evaluate_ensemble(predictions, weights, spreads, covers)
        # Maximize hit rate (minimize negative)
        return -result['hit_rate']

    # Start with equal weights
    w0 = np.ones(n_models) / n_models

    # Constraints: weights must be positive
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    # Optimize
    result = minimize(objective, w0, method='SLSQP', bounds=bounds)

    # Normalize
    optimal_w = result.x / result.x.sum()

    return {name: optimal_w[i] for i, name in enumerate(model_names)}


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

    # Generate predictions for each split
    print("\nGenerating predictions...")

    splits = {
        'val': val_df,
        'test': test_df,
    }

    results = {}

    for split_name, split_df in splits.items():
        predictions = {}

        if 'ridge' in models:
            predictions['ridge'] = predict_ridge(models['ridge'], split_df)
            print(f"  {split_name} Ridge predictions: {len(predictions['ridge'])}")

        if 'gbm' in models:
            predictions['gbm'] = predict_gbm(models['gbm'], split_df)
            print(f"  {split_name} GBM predictions: {len(predictions['gbm'])}")

        if 'dnn' in models:
            predictions['dnn'] = predict_dnn(models['dnn'], split_df)
            print(f"  {split_name} DNN predictions: {len(predictions['dnn'])}")

        results[split_name] = {
            'predictions': predictions,
            'spreads': split_df['spread_a'].values,
            'covers': split_df['cover_a'].values,
            'margins': split_df['final_margin_a'].values,
        }

    # Tune weights on validation set
    print("\n" + "="*60)
    print("TUNING ENSEMBLE WEIGHTS ON VALIDATION SET")
    print("="*60)

    optimal_weights = tune_weights(
        results['val']['predictions'],
        results['val']['spreads'],
        results['val']['covers']
    )

    print("\nOptimal weights:")
    for name, weight in optimal_weights.items():
        print(f"  {name}: {weight:.3f}")

    # Evaluate on validation with optimal weights
    val_result = evaluate_ensemble(
        results['val']['predictions'],
        optimal_weights,
        results['val']['spreads'],
        results['val']['covers']
    )

    print(f"\nValidation ensemble performance:")
    print(f"  Hit Rate: {val_result['hit_rate']:.3f}")
    print(f"  ROI: {val_result['roi']:.3f}")

    # Evaluate on test with optimal weights
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)

    test_result = evaluate_ensemble(
        results['test']['predictions'],
        optimal_weights,
        results['test']['spreads'],
        results['test']['covers']
    )

    print(f"\nTest ensemble performance:")
    print(f"  Hit Rate: {test_result['hit_rate']:.3f}")
    print(f"  ROI: {test_result['roi']:.3f}")
    print(f"  N bets: {test_result['n_bets']}")

    # Compare individual models on test
    print("\nIndividual model performance on test:")
    for name in results['test']['predictions'].keys():
        individual = evaluate_ensemble(
            {name: results['test']['predictions'][name]},
            {name: 1.0},
            results['test']['spreads'],
            results['test']['covers']
        )
        print(f"  {name}: {individual['hit_rate']:.3f} hit rate, {individual['roi']:.3f} ROI")

    # Breakeven check
    breakeven = 0.524
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"Ensemble Hit Rate: {test_result['hit_rate']:.1%}")
    print(f"Breakeven:         {breakeven:.1%}")

    if test_result['hit_rate'] >= breakeven:
        print(f"\n✅ PROFITABLE! {test_result['hit_rate'] - breakeven:.1%} above breakeven")
    else:
        print(f"\n⚠️  {breakeven - test_result['hit_rate']:.1%} below breakeven")

    # Save results
    summary = {
        'ensemble_hit_rate': test_result['hit_rate'],
        'ensemble_roi': test_result['roi'],
        'n_bets': test_result['n_bets'],
        **{f'weight_{k}': v for k, v in optimal_weights.items()},
    }

    pd.DataFrame([summary]).to_csv(metrics_dir / 'ensemble_final_summary.csv', index=False)
    print(f"\n✓ Saved results to {metrics_dir / 'ensemble_final_summary.csv'}")

    # Save ensemble config
    with open(models_dir / 'ensemble_final_config.pkl', 'wb') as f:
        pickle.dump({
            'weights': optimal_weights,
        }, f)


if __name__ == '__main__':
    main()

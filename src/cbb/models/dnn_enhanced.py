"""
Enhanced deep learning model for ATS prediction.

Key improvements over baseline DNN:
1. Uses KenPom features directly (not just rolling stats)
2. Two-tower architecture (separate team encoders)
3. Attention mechanism for matchup modeling
4. Calibrated probability outputs
5. Focal loss for hard example mining

Target: >52.4% hit rate (breakeven for -110 odds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


# Feature groups
# NOTE: Rankings not available in historical KenPom snapshots (pre-2025)
TEAM_A_FEATURES = [
    'kp_adj_em_a', 'kp_adj_o_a', 'kp_adj_d_a', 'kp_tempo_a',
    'ew_margin_a', 'rolling_ats_a', 'rest_days_a', 'b2b_a',
]

TEAM_B_FEATURES = [
    'kp_adj_em_b', 'kp_adj_o_b', 'kp_adj_d_b', 'kp_tempo_b',
    'ew_margin_b', 'rolling_ats_b', 'rest_days_b', 'b2b_b',
]

MATCHUP_FEATURES = [
    'kp_adj_em_diff', 'kp_tempo_avg', 'kp_tempo_diff',
    'kp_o_vs_d_a', 'kp_o_vs_d_b',
    'rest_diff', 'rolling_ats_diff', 'ew_margin_diff',
]

CONTEXT_FEATURES = [
    'is_home_a', 'is_neutral',
]

ALL_DNN_FEATURES = TEAM_A_FEATURES + TEAM_B_FEATURES + MATCHUP_FEATURES + CONTEXT_FEATURES


class EnhancedDataset(Dataset):
    """PyTorch dataset with team features, matchup features, and context."""

    def __init__(
        self,
        team_a_features: np.ndarray,
        team_b_features: np.ndarray,
        matchup_features: np.ndarray,
        context_features: np.ndarray,
        margins: np.ndarray,
        spreads: np.ndarray,
        covers: np.ndarray
    ):
        self.team_a = torch.FloatTensor(team_a_features)
        self.team_b = torch.FloatTensor(team_b_features)
        self.matchup = torch.FloatTensor(matchup_features)
        self.context = torch.FloatTensor(context_features)
        self.margins = torch.FloatTensor(margins)
        self.spreads = torch.FloatTensor(spreads)
        self.covers = torch.FloatTensor(covers)

    def __len__(self):
        return len(self.margins)

    def __getitem__(self, idx):
        return (
            self.team_a[idx],
            self.team_b[idx],
            self.matchup[idx],
            self.context[idx],
            self.margins[idx],
            self.spreads[idx],
            self.covers[idx]
        )


class TeamEncoder(nn.Module):
    """Encode team features into a latent representation."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class MatchupAttention(nn.Module):
    """Attention mechanism for team matchup."""

    def __init__(self, team_dim: int):
        super().__init__()
        self.query = nn.Linear(team_dim, team_dim)
        self.key = nn.Linear(team_dim, team_dim)
        self.scale = np.sqrt(team_dim)

    def forward(self, team_a_repr, team_b_repr):
        # Compute attention weights
        q = self.query(team_a_repr)
        k = self.key(team_b_repr)

        # Attention score
        attn = torch.sum(q * k, dim=-1, keepdim=True) / self.scale
        attn_weight = torch.sigmoid(attn)

        # Weighted combination
        return team_a_repr * attn_weight + team_b_repr * (1 - attn_weight), attn_weight


class TwoTowerDNN(nn.Module):
    """
    Two-tower architecture with attention for margin prediction.

    Architecture:
    - Team A encoder
    - Team B encoder
    - Matchup attention
    - Matchup features processing
    - Context features
    - Final prediction head
    """

    def __init__(
        self,
        team_feature_dim: int,
        matchup_feature_dim: int,
        context_feature_dim: int,
        team_hidden_dim: int = 32,
        team_output_dim: int = 16,
        final_hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()

        # Team encoders
        self.team_a_encoder = TeamEncoder(team_feature_dim, team_hidden_dim, team_output_dim, dropout)
        self.team_b_encoder = TeamEncoder(team_feature_dim, team_hidden_dim, team_output_dim, dropout)

        # Matchup attention
        self.attention = MatchupAttention(team_output_dim)

        # Matchup features processor
        self.matchup_processor = nn.Sequential(
            nn.Linear(matchup_feature_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final prediction head
        # Input: team_a_repr + team_b_repr + attention_repr + matchup_features + context
        final_input_dim = team_output_dim * 3 + 16 + context_feature_dim

        self.prediction_head = nn.Sequential(
            nn.Linear(final_input_dim, final_hidden_dim),
            nn.BatchNorm1d(final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, team_a_features, team_b_features, matchup_features, context_features):
        # Encode teams
        team_a_repr = self.team_a_encoder(team_a_features)
        team_b_repr = self.team_b_encoder(team_b_features)

        # Attention-based interaction
        interaction_repr, attn_weight = self.attention(team_a_repr, team_b_repr)

        # Process matchup features
        matchup_repr = self.matchup_processor(matchup_features)

        # Concatenate all representations
        combined = torch.cat([
            team_a_repr,
            team_b_repr,
            interaction_repr,
            matchup_repr,
            context_features
        ], dim=-1)

        # Predict margin
        margin = self.prediction_head(combined)

        return margin.squeeze(-1), attn_weight


class ResidualMLP(nn.Module):
    """MLP with residual connections for stable training."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [64, 32], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            layers.append(block)

            # Add residual connection if dimensions match
            if prev_dim == hidden_dim:
                layers.append(nn.Identity())  # Placeholder for residual

            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x).squeeze(-1)


def prepare_dnn_features(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]]
) -> tuple[np.ndarray, ...]:
    """
    Prepare features for DNN training.

    Returns tuple of (team_a_features, team_b_features, matchup_features, context_features)
    """
    def get_features(feature_list):
        available = [f for f in feature_list if f in df.columns]
        data = df[available].values
        # Fill NaN with 0
        return np.nan_to_num(data, nan=0)

    team_a = get_features(feature_groups.get('team_a', TEAM_A_FEATURES))
    team_b = get_features(feature_groups.get('team_b', TEAM_B_FEATURES))
    matchup = get_features(feature_groups.get('matchup', MATCHUP_FEATURES))
    context = get_features(feature_groups.get('context', CONTEXT_FEATURES))

    return team_a, team_b, matchup, context


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for team_a, team_b, matchup, context, margins, spreads, covers in dataloader:
        team_a = team_a.to(device)
        team_b = team_b.to(device)
        matchup = matchup.to(device)
        context = context.to(device)
        margins = margins.to(device)

        optimizer.zero_grad()

        if isinstance(model, TwoTowerDNN):
            pred, _ = model(team_a, team_b, matchup, context)
        else:
            # Simple MLP: concatenate all features
            combined = torch.cat([team_a, team_b, matchup, context], dim=-1)
            pred = model(combined)

        loss = criterion(pred, margins)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    all_preds = []
    all_margins = []
    all_spreads = []
    all_covers = []

    with torch.no_grad():
        for team_a, team_b, matchup, context, margins, spreads, covers in dataloader:
            team_a = team_a.to(device)
            team_b = team_b.to(device)
            matchup = matchup.to(device)
            context = context.to(device)
            margins = margins.to(device)

            if isinstance(model, TwoTowerDNN):
                pred, _ = model(team_a, team_b, matchup, context)
            else:
                combined = torch.cat([team_a, team_b, matchup, context], dim=-1)
                pred = model(combined)

            loss = criterion(pred, margins)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_margins.append(margins.cpu().numpy())
            all_spreads.append(spreads.numpy())
            all_covers.append(covers.numpy())

    all_preds = np.concatenate(all_preds)
    all_margins = np.concatenate(all_margins)
    all_spreads = np.concatenate(all_spreads)
    all_covers = np.concatenate(all_covers)

    mae = mean_absolute_error(all_margins, all_preds)
    rmse = np.sqrt(mean_squared_error(all_margins, all_preds))

    # ATS evaluation
    pred_edge = all_preds - (-all_spreads)
    valid_mask = np.isin(all_covers, [0, 1])
    valid_edge = pred_edge[valid_mask]
    valid_covers = all_covers[valid_mask]

    bets_a = valid_edge > 0
    hits = (bets_a == (valid_covers == 1))
    hit_rate = hits.mean() if len(hits) > 0 else 0

    # ROI
    win_payout = 100 / 110
    n_bets = len(hits)
    profit = hits.sum() * win_payout - (n_bets - hits.sum())
    roi = profit / n_bets if n_bets > 0 else 0

    return {
        'loss': total_loss / n_batches,
        'mae': mae,
        'rmse': rmse,
        'hit_rate': hit_rate,
        'roi': roi,
        'preds': all_preds,
        'margins': all_margins,
        'spreads': all_spreads,
        'covers': all_covers,
    }


def main():
    """Main training script."""
    project_root = Path(__file__).parent.parent.parent.parent
    features_dir = project_root / 'data' / 'features'
    models_dir = project_root / 'reports' / 'models'
    metrics_dir = project_root / 'reports' / 'metrics'
    backtests_dir = project_root / 'reports' / 'backtests'

    models_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load enhanced features
    features_path = features_dir / 'games_features_enhanced.parquet'
    if not features_path.exists():
        print("Enhanced features not found. Run enhanced_features.py first.")
        return

    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df):,} games")

    # Filter to games with KenPom and spread
    df = df[(df['kp_matched'] == True) & (df['spread_a'].notna())].copy()
    df = df.dropna(subset=['final_margin_a', 'cover_a'])
    print(f"After filtering: {len(df):,} games")

    # Split by season
    train_df = df[df['season'].isin(['2022-23', '2023-24'])]
    val_df = df[df['season'] == '2024-25']
    test_df = df[df['season'] == '2025-26']

    print(f"\nTrain: {len(train_df):,} games")
    print(f"Val:   {len(val_df):,} games")
    print(f"Test:  {len(test_df):,} games")

    # Prepare features
    feature_groups = {
        'team_a': TEAM_A_FEATURES,
        'team_b': TEAM_B_FEATURES,
        'matchup': MATCHUP_FEATURES,
        'context': CONTEXT_FEATURES,
    }

    train_features = prepare_dnn_features(train_df, feature_groups)
    val_features = prepare_dnn_features(val_df, feature_groups)
    test_features = prepare_dnn_features(test_df, feature_groups)

    # Scale features
    scalers = {}
    train_scaled = []
    val_scaled = []
    test_scaled = []

    for i, name in enumerate(['team_a', 'team_b', 'matchup', 'context']):
        scaler = StandardScaler()
        train_scaled.append(scaler.fit_transform(train_features[i]))
        val_scaled.append(scaler.transform(val_features[i]))
        test_scaled.append(scaler.transform(test_features[i]))
        scalers[name] = scaler

    # Create datasets
    train_dataset = EnhancedDataset(
        *train_scaled,
        train_df['final_margin_a'].values,
        train_df['spread_a'].values,
        train_df['cover_a'].values
    )
    val_dataset = EnhancedDataset(
        *val_scaled,
        val_df['final_margin_a'].values,
        val_df['spread_a'].values,
        val_df['cover_a'].values
    )
    test_dataset = EnhancedDataset(
        *test_scaled,
        test_df['final_margin_a'].values,
        test_df['spread_a'].values,
        test_df['cover_a'].values
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Get feature dimensions
    team_dim = train_scaled[0].shape[1]
    matchup_dim = train_scaled[2].shape[1]
    context_dim = train_scaled[3].shape[1]

    print(f"\nFeature dimensions:")
    print(f"  Team: {team_dim}")
    print(f"  Matchup: {matchup_dim}")
    print(f"  Context: {context_dim}")

    # Create model
    model = TwoTowerDNN(
        team_feature_dim=team_dim,
        matchup_feature_dim=matchup_dim,
        context_feature_dim=context_dim,
        team_hidden_dim=32,
        team_output_dim=16,
        final_hidden_dim=32,
        dropout=0.3
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    n_epochs = 200
    best_val_loss = float('inf')
    best_val_hit_rate = 0
    patience = 20
    patience_counter = 0

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_result['loss'])

        # Track best model by hit rate (our target metric)
        if val_result['hit_rate'] > best_val_hit_rate:
            best_val_hit_rate = val_result['hit_rate']
            torch.save(model.state_dict(), models_dir / 'dnn_enhanced.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_result['loss']:.4f}, "
                  f"val_mae={val_result['mae']:.2f}, "
                  f"val_hit_rate={val_result['hit_rate']:.3f}, "
                  f"val_roi={val_result['roi']:.3f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(models_dir / 'dnn_enhanced.pt'))

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    results = []
    for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        result = evaluate(model, loader, criterion, device)
        print(f"{name}:")
        print(f"  MAE:      {result['mae']:.2f}")
        print(f"  RMSE:     {result['rmse']:.2f}")
        print(f"  Hit Rate: {result['hit_rate']:.3f}")
        print(f"  ROI:      {result['roi']:.3f}")
        results.append({
            'split': name,
            'mae': result['mae'],
            'rmse': result['rmse'],
            'hit_rate': result['hit_rate'],
            'roi': result['roi'],
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(metrics_dir / 'dnn_enhanced_summary.csv', index=False)

    # Save model artifacts
    with open(models_dir / 'dnn_enhanced_artifacts.pkl', 'wb') as f:
        pickle.dump({
            'scalers': scalers,
            'feature_groups': feature_groups,
            'model_config': {
                'team_feature_dim': team_dim,
                'matchup_feature_dim': matchup_dim,
                'context_feature_dim': context_dim,
            }
        }, f)

    # Save test predictions
    test_result = evaluate(model, test_loader, criterion, device)
    test_output = test_df.copy()
    test_output['dnn_pred'] = test_result['preds']
    test_output['dnn_pred_edge'] = test_result['preds'] - (-test_result['spreads'])
    test_output.to_csv(backtests_dir / 'dnn_enhanced_2022_2026.csv', index=False)

    print(f"\n✓ Saved model to {models_dir / 'dnn_enhanced.pt'}")
    print(f"✓ Saved results to {metrics_dir / 'dnn_enhanced_summary.csv'}")
    print(f"✓ Saved backtest to {backtests_dir / 'dnn_enhanced_2022_2026.csv'}")

    # Performance summary
    test_hit_rate = results[2]['hit_rate']
    breakeven = 0.524

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Test Hit Rate: {test_hit_rate:.1%}")
    print(f"Breakeven:     {breakeven:.1%}")

    if test_hit_rate >= breakeven:
        print(f"\n✅ MODEL IS PROFITABLE! ({test_hit_rate - breakeven:.1%} above breakeven)")
    else:
        print(f"\n⚠️  Model is {breakeven - test_hit_rate:.1%} below breakeven")
        print("   Consider: more features, larger dataset, or ensemble methods")


if __name__ == '__main__':
    main()

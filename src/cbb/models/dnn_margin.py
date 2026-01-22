"""
Deep learning margin prediction model.

Architecture options:
1. MLP: Simple feedforward network on concatenated features
2. Two-tower: Separate encoders for Team A and Team B, merged for prediction

Output:
- reports/models/dnn_margin.pt
- reports/backtests/dnn_<config>.csv
- reports/metrics/dnn_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


class MarginDataset(Dataset):
    """PyTorch dataset for margin prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray, spreads: np.ndarray, covers: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.spreads = torch.FloatTensor(spreads)
        self.covers = torch.FloatTensor(covers)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.spreads[idx], self.covers[idx]


class MLPMarginModel(nn.Module):
    """Simple MLP for margin prediction."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [64, 32], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class TwoTowerMarginModel(nn.Module):
    """
    Two-tower architecture for margin prediction.

    Separate encoders for Team A and Team B features,
    merged for final prediction.
    """

    def __init__(
        self,
        team_a_dim: int,
        team_b_dim: int,
        context_dim: int,
        tower_dim: int = 32,
        merge_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()

        # Team A tower
        self.team_a_tower = nn.Sequential(
            nn.Linear(team_a_dim, tower_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Team B tower
        self.team_b_tower = nn.Sequential(
            nn.Linear(team_b_dim, tower_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Merge layer
        merge_input_dim = tower_dim * 2 + context_dim
        self.merge = nn.Sequential(
            nn.Linear(merge_input_dim, merge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(merge_dim, 1),
        )

    def forward(self, team_a_features, team_b_features, context_features):
        team_a_repr = self.team_a_tower(team_a_features)
        team_b_repr = self.team_b_tower(team_b_features)

        merged = torch.cat([team_a_repr, team_b_repr, context_features], dim=-1)
        return self.merge(merged).squeeze(-1)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for X, y, spreads, covers in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
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
    all_targets = []
    all_spreads = []
    all_covers = []

    with torch.no_grad():
        for X, y, spreads, covers in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_spreads.append(spreads.numpy())
            all_covers.append(covers.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_spreads = np.concatenate(all_spreads)
    all_covers = np.concatenate(all_covers)

    mae = mean_absolute_error(all_targets, all_preds)

    # ATS hit rate
    pred_edge = all_preds - (-all_spreads)
    valid_mask = np.isin(all_covers, [0, 1])
    simple_bets = pred_edge[valid_mask] > 0
    simple_hits = (simple_bets == (all_covers[valid_mask] == 1))
    hit_rate = simple_hits.mean() if len(simple_hits) > 0 else 0

    return {
        'loss': total_loss / n_batches,
        'mae': mae,
        'hit_rate': hit_rate,
        'preds': all_preds,
        'targets': all_targets,
        'spreads': all_spreads,
        'covers': all_covers,
    }


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent.parent
    processed_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'reports' / 'models'
    backtests_dir = project_root / 'reports' / 'backtests'
    metrics_dir = project_root / 'reports' / 'metrics'

    models_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import feature preparation from baseline
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
    train_df = features_df[features_df['season'].isin(['2022-23', '2023-24'])]
    val_df = features_df[features_df['season'] == '2024-25']
    test_df = features_df[features_df['season'] == '2025-26']

    print(f"\nTrain: {len(train_df):,} games")
    print(f"Val: {len(val_df):,} games")
    print(f"Test: {len(test_df):,} games")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    y_train = train_df['final_margin_a'].values
    y_val = val_df['final_margin_a'].values
    y_test = test_df['final_margin_a'].values

    spreads_train = train_df['spread_a'].values
    spreads_val = val_df['spread_a'].values
    spreads_test = test_df['spread_a'].values

    covers_train = train_df['cover_a'].values
    covers_val = val_df['cover_a'].values
    covers_test = test_df['cover_a'].values

    # Create datasets
    train_dataset = MarginDataset(X_train, y_train, spreads_train, covers_train)
    val_dataset = MarginDataset(X_val, y_val, spreads_val, covers_val)
    test_dataset = MarginDataset(X_test, y_test, spreads_test, covers_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Create model
    input_dim = len(feature_cols)
    model = MLPMarginModel(input_dim, hidden_dims=[64, 32], dropout=0.2)
    model = model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print("\nTraining DNN...")
    n_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_result['loss'])

        if val_result['loss'] < best_val_loss:
            best_val_loss = val_result['loss']
            torch.save(model.state_dict(), models_dir / 'dnn_margin.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_result['loss']:.4f}, "
                  f"val_mae={val_result['mae']:.2f}, val_hit_rate={val_result['hit_rate']:.3f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(models_dir / 'dnn_margin.pt'))

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    results = []
    for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        result = evaluate(model, loader, criterion, device)
        print(f"{name}: MAE={result['mae']:.2f}, Hit Rate={result['hit_rate']:.3f}")
        results.append({
            'split': name,
            'mae': result['mae'],
            'hit_rate': result['hit_rate'],
        })

    # Save DNN results
    results_df = pd.DataFrame(results)
    results_df.to_csv(metrics_dir / 'dnn_summary.csv', index=False)

    # Compare with baseline
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)

    baseline_df = pd.read_csv(metrics_dir / 'baseline_summary.csv')
    print("\nBaseline Results:")
    print(baseline_df.to_string(index=False))

    print("\nDNN Results:")
    print(results_df.to_string(index=False))

    # Save test predictions
    test_result = evaluate(model, test_loader, criterion, device)
    test_output = test_df.copy()
    test_output['dnn_pred'] = test_result['preds']
    test_output['dnn_pred_edge'] = test_result['preds'] - (-test_result['spreads'])
    test_output.to_csv(backtests_dir / 'dnn_mlp_2022_2026.csv', index=False)

    # Save model artifacts
    with open(models_dir / 'dnn_margin_artifacts.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_cols': feature_cols,
            'model_config': {'input_dim': input_dim, 'hidden_dims': [64, 32], 'dropout': 0.2}
        }, f)

    print(f"\n✓ Saved model to {models_dir / 'dnn_margin.pt'}")
    print(f"✓ Saved results to {metrics_dir / 'dnn_summary.csv'}")
    print(f"✓ Saved backtest to {backtests_dir / 'dnn_mlp_2022_2026.csv'}")


if __name__ == '__main__':
    main()

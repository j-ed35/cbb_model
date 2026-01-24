"""
Full training pipeline v2 with Ridge, GBM, and DNN.

Compares baseline features vs enhanced v2 features with DNN ensemble.
Uses extended KenPom features including Four Factors, HCA, and Height/Exp.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.cbb.utils.evaluation import evaluate_ats


# Feature groups for v2
KENPOM_CORE = [
    "kp_adj_em_a", "kp_adj_em_b", "kp_adj_em_diff",
    "kp_adj_o_a", "kp_adj_o_b",
    "kp_adj_d_a", "kp_adj_d_b",
    "kp_tempo_a", "kp_tempo_b", "kp_tempo_avg", "kp_tempo_diff",
    "kp_o_vs_d_a", "kp_o_vs_d_b",
]

SITUATIONAL = ["rest_days_a", "rest_days_b", "rest_diff", "b2b_a", "b2b_b"]
ROLLING = ["rolling_ats_a", "rolling_ats_b", "rolling_ats_diff"]
RECENCY = ["ew_margin_a", "ew_margin_b", "ew_margin_diff"]
CONTEXT = ["is_home_a", "is_neutral"]

# NEW features from extended KenPom
HCA = ["team_hca"]
FOUR_FACTORS_MATCHUP = [
    "ff_efg_matchup_a", "ff_efg_matchup_b",
    "ff_to_matchup_a", "ff_to_matchup_b",
    "ff_reb_matchup_a", "ff_reb_matchup_b",
]
HEIGHT_EXP = [
    "ht_height_diff", "ht_exp_diff", "ht_cont_diff",
]

BASELINE_FEATURES = KENPOM_CORE + SITUATIONAL + ROLLING + RECENCY + CONTEXT
V2_FEATURES = BASELINE_FEATURES + HCA + FOUR_FACTORS_MATCHUP + HEIGHT_EXP


class SimpleDNN(nn.Module):
    """Simple feedforward DNN for margin prediction."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
) -> SimpleDNN:
    """Train DNN model."""
    device = torch.device("cpu")  # Keep on CPU for compatibility

    model = SimpleDNN(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    project_root: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Run full training and evaluation."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    # Prepare data
    available = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=available + ["spread_a", "final_margin_a"])
    df_clean = df_clean[df_clean["kp_matched"] == True]

    train = df_clean[df_clean["season"] != "2025-26"]
    test = df_clean[df_clean["season"] == "2025-26"]

    print(f"Features: {len(available)}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    X_train = train[available].values
    y_train = train["final_margin_a"].values
    X_test = test[available].values
    y_test = test["final_margin_a"].values
    spreads_test = test["spread_a"].values
    covers_test = test["cover_a"].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split train into train/val for DNN
    val_size = int(len(X_train) * 0.15)
    X_tr, X_val = X_train_scaled[:-val_size], X_train_scaled[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    # Train models
    print("Training Ridge...", end=" ")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    print(f"MAE: {mean_absolute_error(y_test, ridge_pred):.2f}")

    print("Training GBM...", end=" ")
    gbm = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, gbm_pred):.2f}")

    print("Training DNN...", end=" ")
    dnn = train_dnn(X_tr, y_tr, X_val, y_val)
    dnn.eval()
    with torch.no_grad():
        dnn_pred = dnn(torch.FloatTensor(X_test_scaled)).numpy()
    print(f"MAE: {mean_absolute_error(y_test, dnn_pred):.2f}")

    # Ensemble: 10% Ridge, 20% GBM, 70% DNN (as per original)
    ensemble_pred = 0.1 * ridge_pred + 0.2 * gbm_pred + 0.7 * dnn_pred
    print(f"Ensemble MAE: {mean_absolute_error(y_test, ensemble_pred):.2f}")

    # Evaluate using shared utility
    print("\nATS Performance (Test Set 2025-26):")
    print(f"{'Threshold':<12} {'Hit Rate':<12} {'ROI':<12} {'N Bets'}")
    print("-" * 48)

    results = {}
    for thresh in [0.0, 4.0, 4.5, 5.0, 6.0]:
        ats = evaluate_ats(
            predictions=ensemble_pred,
            spreads=spreads_test,
            covers=covers_test,
            threshold=thresh,
        )
        results[thresh] = ats
        status = " " if ats["hit_rate"] >= 0.524 else " "
        print(
            f"{thresh:<12} {ats['hit_rate']*100:.1f}%{status:<6} "
            f"{ats['roi']*100:+.1f}%{'':<6} {ats['n_bets']}"
        )

    if save:
        models_dir = project_root / "reports" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_data = {
            "ridge": ridge,
            "gbm": gbm,
            "scaler": scaler,
            "feature_cols": available,
            "weights": (0.1, 0.2, 0.7),
            "version": "v2_full",
        }

        with open(models_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model_data, f)

        torch.save(dnn.state_dict(), models_dir / f"{name}_dnn.pt")

        with open(models_dir / f"{name}_dnn_config.pkl", "wb") as f:
            pickle.dump({"input_dim": len(available)}, f)

        print(f"\nSaved to {models_dir}/{name}.*")

    return {
        "name": name,
        "n_features": len(available),
        "mae": mean_absolute_error(y_test, ensemble_pred),
        "results": results,
    }


def main() -> None:
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent

    print("Loading v2 features...")
    v2_path = project_root / "data" / "features" / "games_features_v2.parquet"

    if not v2_path.exists():
        print(f"V2 features not found at {v2_path}")
        print("Run build_features_v2.py first to generate extended features.")
        return

    df = pd.read_parquet(v2_path)
    print(f"Total games: {len(df)}")

    # Run experiments
    baseline = run_experiment(df, BASELINE_FEATURES, "baseline_full", project_root)
    v2 = run_experiment(df, V2_FEATURES, "enhanced_v2_full", project_root, save=True)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Features':<10} {'MAE':<8} {'Hit@4.5':<10} {'ROI@4.5'}")
    print("-" * 65)

    for exp in [baseline, v2]:
        ats = exp["results"][4.5]
        status = " " if ats["hit_rate"] >= 0.524 else ""
        print(
            f"{exp['name']:<25} {exp['n_features']:<10} {exp['mae']:.2f}{'':<4} "
            f"{ats['hit_rate']*100:.1f}%{status:<5} {ats['roi']*100:+.1f}%"
        )

    # Improvement
    b_hr = baseline["results"][4.5]["hit_rate"]
    v_hr = v2["results"][4.5]["hit_rate"]
    b_roi = baseline["results"][4.5]["roi"]
    v_roi = v2["results"][4.5]["roi"]

    print(f"\nImprovement: {(v_hr-b_hr)*100:+.1f}pp hit rate, {(v_roi-b_roi)*100:+.1f}pp ROI")

    if v_hr >= 0.524:
        print("\n V2 model achieves breakeven threshold!")
    else:
        print(f"\n V2 model at {v_hr*100:.1f}%, need {52.4:.1f}% for breakeven")


if __name__ == "__main__":
    main()

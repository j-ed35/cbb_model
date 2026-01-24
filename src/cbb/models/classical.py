"""Classical ML models (Ridge + GBM) for margin prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def train_ridge_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ridge_alpha: float = 1.0,
    gbm_params: Optional[dict] = None,
) -> tuple[Ridge, GradientBoostingRegressor, StandardScaler]:
    """
    Train Ridge and GBM models.

    Returns:
        (ridge_model, gbm_model, scaler)
    """
    if gbm_params is None:
        gbm_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "min_samples_leaf": 20,
            "random_state": 42,
        }

    # Scale for Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train Ridge
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_scaled, y_train)

    # Train GBM (uses raw features)
    gbm = GradientBoostingRegressor(**gbm_params)
    gbm.fit(X_train, y_train)

    return ridge, gbm, scaler

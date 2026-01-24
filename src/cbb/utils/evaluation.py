"""
Centralized ATS evaluation functions.

This module provides a single source of truth for all ATS (Against The Spread)
evaluation metrics used across training, backtesting, and prediction.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def compute_roi(hits: np.ndarray, juice: float = -110) -> float:
    """
    Compute ROI assuming standard juice.

    Args:
        hits: Boolean array of whether each bet won
        juice: Vig/juice (default -110)

    Returns:
        ROI as a decimal (0.05 = 5% profit)
    """
    if len(hits) == 0:
        return 0.0

    win_payout = 100 / abs(juice)
    total_profit = hits.sum() * win_payout - (~hits).sum() * 1.0
    total_risked = len(hits)

    return total_profit / total_risked if total_risked > 0 else 0.0


def evaluate_ats(
    predictions: np.ndarray,
    spreads: np.ndarray,
    margins: Optional[np.ndarray] = None,
    covers: Optional[np.ndarray] = None,
    threshold: float = 0.0,
    juice: float = -110,
) -> dict:
    """
    Evaluate ATS (Against The Spread) performance.

    This is the canonical ATS evaluation function. Use this instead of
    implementing your own evaluation logic.

    Args:
        predictions: Predicted margins from team A perspective
        spreads: Point spreads from team A perspective (negative = favorite)
        margins: Actual game margins (optional if covers provided)
        covers: Actual cover results (1=cover, 0=no cover, 0.5=push)
                If not provided, computed from margins and spreads
        threshold: Minimum absolute edge to place bet (0 = bet all games)
        juice: Vig/juice for ROI calculation (default -110)

    Returns:
        Dictionary with:
            - hit_rate: Fraction of bets won
            - roi: Return on investment
            - n_bets: Number of bets placed
            - n_games: Total games evaluated
            - mae: Mean absolute error (if margins provided)
            - threshold: Edge threshold used
    """
    # Calculate predicted edge: how much better we think Team A is vs the spread
    # If spread is -5 (Team A favored by 5), implied margin is +5
    # Edge = prediction - implied_margin
    pred_edge = predictions - (-spreads)

    # Compute covers from margins if not provided
    if covers is None:
        if margins is None:
            raise ValueError("Must provide either margins or covers")
        # Cover if actual margin beats the spread
        margin_vs_spread = margins + spreads  # margins - (-spreads)
        covers = np.where(
            margin_vs_spread > 0, 1.0,
            np.where(margin_vs_spread < 0, 0.0, 0.5)
        )

    # Filter valid covers (exclude pushes)
    valid_mask = np.isin(covers, [0, 1])
    valid_edge = pred_edge[valid_mask]
    valid_covers = covers[valid_mask].astype(int)

    # Apply threshold - only bet when |edge| >= threshold
    bet_mask = np.abs(valid_edge) >= threshold

    result = {
        "n_games": len(predictions),
        "threshold": threshold,
    }

    # Add MAE if margins provided
    if margins is not None:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        result["mae"] = mean_absolute_error(margins, predictions)
        result["rmse"] = np.sqrt(mean_squared_error(margins, predictions))

    if bet_mask.sum() == 0:
        result.update({
            "hit_rate": 0.0,
            "roi": 0.0,
            "n_bets": 0,
            "pct_games_bet": 0.0,
        })
        return result

    filtered_edge = valid_edge[bet_mask]
    filtered_covers = valid_covers[bet_mask]

    # Betting logic: bet on Team A covering if edge > 0, else bet Team B covers
    bets_a = filtered_edge > 0
    hits = (bets_a == (filtered_covers == 1))

    hit_rate = hits.mean()
    roi = compute_roi(hits, juice)

    result.update({
        "hit_rate": float(hit_rate),
        "roi": float(roi),
        "n_bets": int(bet_mask.sum()),
        "pct_games_bet": float(bet_mask.sum() / len(valid_mask)) if len(valid_mask) > 0 else 0.0,
    })

    return result


def evaluate_with_threshold(
    predictions: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    threshold: float,
    juice: float = -110,
) -> dict:
    """
    Evaluate ATS with a specific edge threshold.

    This is a convenience wrapper around evaluate_ats for threshold-specific evaluation.

    Args:
        predictions: Predicted margins
        spreads: Point spreads
        covers: Actual cover results
        threshold: Minimum absolute edge to bet
        juice: Vig for ROI calculation

    Returns:
        Dictionary with hit_rate, roi, n_bets, threshold
    """
    return evaluate_ats(
        predictions=predictions,
        spreads=spreads,
        covers=covers,
        threshold=threshold,
        juice=juice,
    )


def tune_threshold(
    predictions: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    min_bets: int = 50,
    thresholds: Optional[list[float]] = None,
) -> tuple[float, dict]:
    """
    Find optimal edge threshold on validation set.

    Args:
        predictions: Predicted margins
        spreads: Point spreads
        covers: Actual cover results
        min_bets: Minimum number of bets required for a valid threshold
        thresholds: List of thresholds to try (default: 0 to 7 in 0.5 steps)

    Returns:
        Tuple of (best_threshold, best_metrics)
    """
    if thresholds is None:
        thresholds = [i * 0.5 for i in range(15)]  # 0.0 to 7.0

    best_roi = -np.inf
    best_threshold = 0.0
    best_metrics = None

    for threshold in thresholds:
        metrics = evaluate_ats(
            predictions=predictions,
            spreads=spreads,
            covers=covers,
            threshold=threshold,
        )
        if metrics["n_bets"] >= min_bets and metrics["roi"] > best_roi:
            best_roi = metrics["roi"]
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = evaluate_ats(predictions, spreads, covers, threshold=0.0)

    return best_threshold, best_metrics


def analyze_by_edge_bucket(
    predictions: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    buckets: Optional[list[tuple[float, float, str]]] = None,
) -> list[dict]:
    """
    Analyze performance by edge magnitude buckets.

    Args:
        predictions: Predicted margins
        spreads: Point spreads
        covers: Actual cover results
        buckets: List of (low, high, label) tuples defining buckets

    Returns:
        List of dictionaries with bucket analysis
    """
    if buckets is None:
        buckets = [
            (0, 1, "0-1 pts"),
            (1, 2, "1-2 pts"),
            (2, 3, "2-3 pts"),
            (3, 4, "3-4 pts"),
            (4, 5, "4-5 pts"),
            (5, 7, "5-7 pts"),
            (7, 10, "7-10 pts"),
            (10, float("inf"), "10+ pts"),
        ]

    pred_edge = predictions - (-spreads)
    abs_edge = np.abs(pred_edge)

    # Filter valid covers
    valid_mask = np.isin(covers, [0, 1])
    valid_edge = pred_edge[valid_mask]
    valid_abs_edge = abs_edge[valid_mask]
    valid_covers = covers[valid_mask].astype(int)

    results = []
    for low, high, label in buckets:
        mask = (valid_abs_edge >= low) & (valid_abs_edge < high)
        if mask.sum() == 0:
            continue

        bucket_edge = valid_edge[mask]
        bucket_covers = valid_covers[mask]

        bets_a = bucket_edge > 0
        hits = (bets_a == (bucket_covers == 1))

        results.append({
            "bucket": label,
            "n_games": int(mask.sum()),
            "hit_rate": float(hits.mean()),
            "roi": float(compute_roi(hits)),
        })

    return results

"""
Calibration utilities for ATS betting.

Addresses OPTIMIZATION_REPORT.md failure points:
1. Edge not monotonic with profitability (lines 173-174)
2. DNN overconfidence (lines 39-43)
3. Fixed ensemble weights suboptimal (line 116)
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import Optional
import pickle
from pathlib import Path


class EdgeCalibrator:
    """
    Calibrates raw edge to cover probability using isotonic regression.

    Addresses: "Edge not monotonic with profitability" (report line 173-174)
    """

    def __init__(self):
        self.calibrator: Optional[IsotonicRegression] = None
        self.fitted = False

    def fit(self, edges: np.ndarray, covers: np.ndarray) -> "EdgeCalibrator":
        """
        Fit calibrator on validation data.

        Args:
            edges: Predicted edges (pred - (-spread))
            covers: Actual cover results (1=cover, 0=no cover)
        """
        # Filter valid covers
        valid_mask = np.isin(covers, [0, 1])
        valid_edges = edges[valid_mask]
        valid_covers = covers[valid_mask].astype(int)

        # For betting on team A when edge > 0, we want P(cover | edge)
        # Use absolute edge and whether the bet direction was correct
        bets_a = valid_edges > 0
        hits = (bets_a == (valid_covers == 1)).astype(int)
        abs_edges = np.abs(valid_edges)

        self.calibrator = IsotonicRegression(y_min=0.4, y_max=0.7, out_of_bounds='clip')
        self.calibrator.fit(abs_edges, hits)
        self.fitted = True

        return self

    def predict_proba(self, edges: np.ndarray) -> np.ndarray:
        """Return calibrated probability of correct bet."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        return self.calibrator.predict(np.abs(edges))

    def get_bet_mask(self, edges: np.ndarray, min_prob: float = 0.535) -> np.ndarray:
        """Return mask for bets where calibrated probability exceeds threshold."""
        probs = self.predict_proba(edges)
        return probs >= min_prob


class BucketSelector:
    """
    Implements bucket-selective betting based on validation performance.

    Addresses: "5-7pt bucket performs best; 7-10 and 10+ underperform" (report line 174)
    """

    def __init__(self):
        self.profitable_buckets: list[tuple[float, float]] = []
        self.bucket_stats: dict = {}

    def fit(
        self,
        edges: np.ndarray,
        covers: np.ndarray,
        breakeven: float = 0.524,
        min_samples: int = 30,
    ) -> "BucketSelector":
        """
        Identify profitable edge buckets on validation data.

        Args:
            edges: Predicted edges
            covers: Actual cover results
            breakeven: Hit rate needed for profitability
            min_samples: Minimum samples per bucket
        """
        buckets = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 7), (7, 10), (10, float('inf'))
        ]

        valid_mask = np.isin(covers, [0, 1])
        valid_edges = edges[valid_mask]
        valid_covers = covers[valid_mask].astype(int)
        abs_edges = np.abs(valid_edges)

        bets_a = valid_edges > 0
        hits = (bets_a == (valid_covers == 1))

        self.profitable_buckets = []
        self.bucket_stats = {}

        for low, high in buckets:
            mask = (abs_edges >= low) & (abs_edges < high)
            n = mask.sum()

            if n >= min_samples:
                bucket_hits = hits[mask]
                hit_rate = bucket_hits.mean()

                self.bucket_stats[(low, high)] = {
                    'n': int(n),
                    'hit_rate': float(hit_rate),
                    'profitable': hit_rate >= breakeven
                }

                if hit_rate >= breakeven:
                    self.profitable_buckets.append((low, high))

        return self

    def get_bet_mask(self, edges: np.ndarray) -> np.ndarray:
        """Return mask for edges in profitable buckets."""
        abs_edges = np.abs(edges)
        mask = np.zeros(len(edges), dtype=bool)

        for low, high in self.profitable_buckets:
            mask |= (abs_edges >= low) & (abs_edges < high)

        return mask


class ROIOptimizedWeights:
    """
    Optimize ensemble weights for ROI instead of hit rate.

    Addresses: "Fixed weights may be suboptimal" (report line 116)
    """

    def __init__(self):
        self.weights: dict[str, float] = {'ridge': 0.1, 'gbm': 0.2, 'dnn': 0.7}

    def fit(
        self,
        predictions: dict[str, np.ndarray],
        spreads: np.ndarray,
        covers: np.ndarray,
        n_trials: int = 100,
    ) -> "ROIOptimizedWeights":
        """
        Grid search for weights that maximize ROI on validation.

        Args:
            predictions: Dict of model_name -> predictions
            spreads: Point spreads
            covers: Actual cover results
            n_trials: Number of random weight combinations to try
        """
        from src.cbb.utils.evaluation import evaluate_ats

        valid_mask = np.isin(covers, [0, 1])
        model_names = list(predictions.keys())

        best_roi = -np.inf
        best_weights = self.weights.copy()

        # Grid search over weight combinations
        np.random.seed(42)
        for _ in range(n_trials):
            # Random weights that sum to 1
            raw = np.random.dirichlet(np.ones(len(model_names)))
            weights = dict(zip(model_names, raw))

            # Compute ensemble
            ensemble = np.zeros(len(spreads))
            for name, w in weights.items():
                ensemble += w * predictions[name]

            # Evaluate ROI
            result = evaluate_ats(ensemble, spreads, covers=covers, threshold=4.5)

            if result['roi'] > best_roi and result['n_bets'] >= 50:
                best_roi = result['roi']
                best_weights = weights

        self.weights = best_weights
        return self

    def combine(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using optimized weights."""
        ensemble = np.zeros(len(list(predictions.values())[0]))
        total_weight = 0

        for name, preds in predictions.items():
            w = self.weights.get(name, 0)
            ensemble += w * preds
            total_weight += w

        if total_weight > 0:
            ensemble /= total_weight
            ensemble *= sum(self.weights.get(n, 0) for n in predictions)

        return ensemble


def temperature_scale_predictions(
    predictions: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    temperature_range: tuple[float, float] = (0.8, 1.5),
) -> tuple[float, np.ndarray]:
    """
    Apply temperature scaling to reduce overconfidence on extreme edges.

    Addresses: "Extreme edges underperform" (report lines 39-43)

    The key insight: edges 10+ underperform mid-range (5-7pt).
    Temperature > 1 shrinks edges, but we need to be careful not to
    over-correct. Default to T=1.0 if no improvement found.

    Args:
        predictions: Raw margin predictions
        spreads: Point spreads
        covers: Actual cover results
        temperature_range: Range of temperatures to search

    Returns:
        Tuple of (optimal_temperature, scaled_predictions)
    """
    from src.cbb.utils.evaluation import evaluate_ats

    # Compute raw edge
    baseline_edge = predictions - (-spreads)

    best_temp = 1.0  # Default: no scaling
    best_hit_rate = 0.0

    # Evaluate baseline first
    baseline_result = evaluate_ats(predictions, spreads, covers=covers, threshold=4.5)
    best_hit_rate = baseline_result['hit_rate']

    # Search for optimal temperature - only use if it improves hit rate
    for temp in np.linspace(temperature_range[0], temperature_range[1], 15):
        if temp == 1.0:
            continue

        # Scale predictions: temp > 1 shrinks edge, temp < 1 expands edge
        scaled_preds = -spreads + baseline_edge / temp

        result = evaluate_ats(scaled_preds, spreads, covers=covers, threshold=4.5)

        # Require improvement in hit rate AND reasonable bet count
        if result['hit_rate'] > best_hit_rate and result['n_bets'] >= 100:
            best_hit_rate = result['hit_rate']
            best_temp = temp

    # Apply best temperature
    scaled_predictions = -spreads + baseline_edge / best_temp

    return best_temp, scaled_predictions


def bootstrap_threshold_stability(
    predictions: np.ndarray,
    spreads: np.ndarray,
    covers: np.ndarray,
    n_bootstrap: int = 100,
    thresholds: list[float] = None,
) -> dict:
    """
    Bootstrap validation to find stable threshold range.

    Addresses: "Avoid single 'best' threshold that doesn't generalize" (report line 188)

    Returns:
        Dict with stable threshold range and confidence intervals
    """
    from src.cbb.utils.evaluation import evaluate_ats

    if thresholds is None:
        thresholds = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    valid_mask = np.isin(covers, [0, 1])
    n_valid = valid_mask.sum()

    # Bootstrap ROIs for each threshold
    threshold_rois = {t: [] for t in thresholds}

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n_valid, size=n_valid, replace=True)

        boot_preds = predictions[valid_mask][idx]
        boot_spreads = spreads[valid_mask][idx]
        boot_covers = covers[valid_mask][idx]

        for thresh in thresholds:
            result = evaluate_ats(boot_preds, boot_spreads, covers=boot_covers, threshold=thresh)
            if result['n_bets'] >= 20:
                threshold_rois[thresh].append(result['roi'])

    # Find most stable profitable threshold
    stability_scores = {}
    for thresh, rois in threshold_rois.items():
        if len(rois) >= 50:
            mean_roi = np.mean(rois)
            std_roi = np.std(rois)
            pct_profitable = np.mean([r > 0 for r in rois])

            stability_scores[thresh] = {
                'mean_roi': mean_roi,
                'std_roi': std_roi,
                'pct_profitable': pct_profitable,
                'stability': mean_roi / (std_roi + 0.01) if pct_profitable > 0.5 else -1
            }

    # Select threshold with best stability (high ROI, low variance, often profitable)
    best_thresh = max(stability_scores.keys(), key=lambda t: stability_scores[t]['stability'])

    return {
        'recommended_threshold': best_thresh,
        'threshold_stats': stability_scores,
    }


class CalibratedBettingSystem:
    """
    Combined calibration system addressing all failure points.

    Usage:
        system = CalibratedBettingSystem()
        system.fit(val_predictions, val_spreads, val_covers)
        bet_mask = system.get_bets(test_predictions, test_spreads)
    """

    def __init__(self):
        self.edge_calibrator = EdgeCalibrator()
        self.bucket_selector = BucketSelector()
        self.weight_optimizer = ROIOptimizedWeights()
        self.temperature = 1.0
        self.stable_threshold = 4.5
        self.fitted = False

    def fit(
        self,
        model_predictions: dict[str, np.ndarray],
        spreads: np.ndarray,
        covers: np.ndarray,
    ) -> "CalibratedBettingSystem":
        """Fit all calibration components on validation data."""

        # 1. Optimize ensemble weights for ROI
        self.weight_optimizer.fit(model_predictions, spreads, covers)

        # 2. Get ensemble predictions with optimized weights
        ensemble = self.weight_optimizer.combine(model_predictions)
        edges = ensemble - (-spreads)

        # 3. Temperature scaling (use original ensemble for evaluation)
        self.temperature, _ = temperature_scale_predictions(
            ensemble, spreads, covers
        )

        # 4. Fit edge calibrator on original edges (not scaled)
        self.edge_calibrator.fit(edges, covers)

        # 5. Fit bucket selector on original edges
        self.bucket_selector.fit(edges, covers)

        # 6. Bootstrap threshold stability on original ensemble
        stability = bootstrap_threshold_stability(ensemble, spreads, covers)
        self.stable_threshold = stability['recommended_threshold']

        self.fitted = True
        return self

    def get_ensemble_predictions(self, model_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Get calibrated ensemble predictions."""
        if not self.fitted:
            # Default behavior
            weights = {'ridge': 0.1, 'gbm': 0.2, 'dnn': 0.7}
            ensemble = np.zeros(len(list(model_predictions.values())[0]))
            for name, preds in model_predictions.items():
                ensemble += weights.get(name, 0) * preds
            return ensemble

        # Use optimized weights
        ensemble = self.weight_optimizer.combine(model_predictions)
        return ensemble

    def get_scaled_predictions(
        self,
        model_predictions: dict[str, np.ndarray],
        spreads: np.ndarray
    ) -> np.ndarray:
        """Get temperature-scaled ensemble predictions."""
        ensemble = self.get_ensemble_predictions(model_predictions)
        edges = ensemble - (-spreads)
        scaled_predictions = -spreads + edges / self.temperature
        return scaled_predictions

    def get_bet_mask(
        self,
        model_predictions: dict[str, np.ndarray],
        spreads: np.ndarray,
        use_bucket_filter: bool = True,
        use_prob_filter: bool = True,
        min_prob: float = 0.535,
    ) -> np.ndarray:
        """
        Get betting mask using all calibration methods.

        Args:
            model_predictions: Dict of model_name -> predictions
            spreads: Point spreads
            use_bucket_filter: Apply bucket-selective betting
            use_prob_filter: Apply probability calibration filter
            min_prob: Minimum calibrated probability

        Returns:
            Boolean mask for which games to bet
        """
        scaled_preds = self.get_scaled_predictions(model_predictions, spreads)
        edges = scaled_preds - (-spreads)

        # Start with threshold filter
        mask = np.abs(edges) >= self.stable_threshold

        # Apply bucket filter
        if use_bucket_filter and self.bucket_selector.profitable_buckets:
            bucket_mask = self.bucket_selector.get_bet_mask(edges)
            mask &= bucket_mask

        # Apply probability filter
        if use_prob_filter and self.edge_calibrator.fitted:
            prob_mask = self.edge_calibrator.get_bet_mask(edges, min_prob)
            mask &= prob_mask

        return mask

    def save(self, path: Path) -> None:
        """Save calibration system to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'edge_calibrator': self.edge_calibrator,
                'bucket_selector': self.bucket_selector,
                'weight_optimizer': self.weight_optimizer,
                'temperature': self.temperature,
                'stable_threshold': self.stable_threshold,
                'fitted': self.fitted,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "CalibratedBettingSystem":
        """Load calibration system from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        system = cls()
        system.edge_calibrator = data['edge_calibrator']
        system.bucket_selector = data['bucket_selector']
        system.weight_optimizer = data['weight_optimizer']
        system.temperature = data['temperature']
        system.stable_threshold = data['stable_threshold']
        system.fitted = data['fitted']

        return system

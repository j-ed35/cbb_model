# CBB Model — Status & Next Steps

## Current Model Performance (after optimization)

**Model**: GBM (default params: n_est=200, depth=4, lr=0.05)
**Shrinkage**: 0.90 | **Threshold**: 1.0 | **Edge Cap**: 2.0

```
Val:  51.6% hit rate, -1.5% ROI (2,820 games)
Test: 49.6% hit rate overall
Test: 51.9% hit rate, -1.0% ROI in betting window (threshold=1.0, cap=2.0, 401 bets)
```

**Edge bucket analysis (test set)**:
```
0-1 pts: 53.4% hit rate (406 games) ROI=+2.0%  ← PROFITABLE
1-2 pts: 51.9% hit rate (401 games) ROI=-1.0%  ← NEAR BREAKEVEN
2-3 pts: 42.6% hit rate (333 games) ROI=-18.6% ← AVOID
3+ pts:  <50.3% across all buckets            ← AVOID
```

## What Was Done

### Phase 1: Feature Expansion & Infrastructure
- Expanded feature set from 15 → 25 (tournament context, rest days, recency margins, rolling ATS)
- Added prediction shrinkage to reduce overconfident predictions
- Added hyperparameter tuning infrastructure (`--tune` flag)
- Added bootstrap threshold tuning for robustness

### Phase 2: Edge Cap Strategy
- Edge bucket analysis proved small edges outperform large edges
- Added `edge_cap` parameter throughout evaluation → training → prediction pipeline
- Model now only bets when `threshold <= |edge| <= edge_cap`

### Phase 3: Neutral Site Fix & Final Training
- **Root cause found**: Raw data had near-zero neutral site flags → `is_home_a`/`is_neutral` were constant → model had no HCA signal
- **Fix**: Override `is_neutral=1` for conference tournament (March 1-16) and postseason (March 17+) games
- Result: 2,825 neutral games (10.2%) instead of ~0 → features now have meaningful signal
- Retrained with all 25 features, default params, fixed neutral detection

## Remaining Work

### High Priority
- **Investigate `ew_margin_diff` dominance** (48.3% importance): verify no data leakage, try training without it
- **More neutral site data**: current heuristic labels ALL March games as neutral, but early-round conference tourneys may be at campus sites

### Medium Priority
- **XGBoost**: Better regularization than sklearn GBM
- **Four Factors features**: Data exists in `data/kenpom_extended/` but not integrated
- **Team-specific HCA**: `build_features_v2.py` has `merge_hca_vectorized()` to replace fixed 3.5 HCA

### Lower Priority
- **Ensemble model**: `ensemble_final.py` combines Ridge + GBM + DNN (currently unused)
- **Live recency features**: `ew_margin_*` and `rolling_ats_*` default to neutral values at inference since live game-by-game data isn't fetched

## Key Commands
```bash
python -m src.cbb.ingest.build_games              # Rebuild base games
python -m src.cbb.features.enhanced_features       # Rebuild features
python -m src.cbb.train_enhanced                   # Train (default params)
python -m src.cbb.train_enhanced --tune            # Train with grid search
python -m src.cbb.predict_daily --json             # Generate daily picks
python -m src.cbb.predict_daily --json --edge-cap 2.5  # Override edge cap
```

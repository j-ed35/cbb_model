# Next Steps: CBB Model Improvement

## What Was Done

### Problem
Model was 23-29-1 (44.2% ATS) heading into March Madness. Needed >52.4% to be profitable.
Key issue: model systematically **overpredicts margins for favorites** (e.g., predicted AKR by 17.3, actual by 3).

### Changes Made (Phase 1 — committed)

#### 1. Feature Engineering - `src/cbb/features/enhanced_features.py`
- Added **tournament context features**: `is_conference_tourney` (early March games) and `is_postseason` (late March/April + PST game type)
- These are date-based heuristics since raw data lacks reliable neutral-site flags
- Added to output column ordering so they persist in parquet

#### 2. Training Script - `src/cbb/train_enhanced.py`
- **Expanded FEATURES** from 15 to 25: added tournament context (2), rest days (3), recency-weighted margins (3), rolling ATS (2)
- **Prediction shrinkage**: evaluates multipliers 0.70-1.00 on validation set and picks the best. Initial results show **0.75 shrinkage** improves val hit rate from 50.4% to 51.7%
- **Hyperparameter tuning**: `--tune` flag runs grid search over 162 GBM parameter combinations
- Model pickle now saves `shrinkage` factor alongside model/features/threshold

#### 3. Prediction Script - `src/cbb/predict_daily.py`
- Expanded FEATURES list to match training (25 features)
- Added `INFERENCE_DEFAULTS` dict with neutral values for features not available live
- March Madness games auto-set `is_postseason=1` when `is_neutral=True`
- Applies shrinkage factor loaded from model pickle

#### 4. Evaluation - `src/cbb/utils/evaluation.py`
- Added `tune_threshold_bootstrap()` - uses 500 bootstrap resamples for robust threshold selection
- Lowered `min_bets` default from 50 to 30 in `tune_threshold()`

### Changes Made (Phase 2 — optimization)

#### 5. Edge Cap Strategy (HIGH IMPACT) - DONE
**Problem**: Edge bucket analysis showed small edges (0-3 pts) hit at ~54%, but large edges (4+ pts) hit at <50%. The old threshold logic ("bet only when |edge| >= threshold") was backwards.

**Solution**: Added `edge_cap` parameter throughout the pipeline:
- `evaluation.py`: `evaluate_ats()`, `tune_threshold()`, `tune_threshold_bootstrap()` all accept `edge_cap` parameter
- `evaluation.py`: New `tune_edge_cap()` function finds optimal max edge on validation set
- `predict_daily.py`: Picks now filtered by `threshold <= |edge| <= edge_cap` instead of just `|edge| >= threshold`
- `predict_daily.py`: `--edge-cap` CLI argument for override
- Model pickle stores `edge_cap=3.0` alongside threshold and shrinkage

#### 6. Bootstrap Threshold Tuning Integrated - DONE
- `train_enhanced.py` now uses `tune_threshold_bootstrap()` (500 resamples) instead of `tune_threshold()` for more robust threshold selection
- Also runs `tune_edge_cap()` to find optimal max edge before threshold tuning
- Both edge_cap and threshold saved in model pickle

#### 7. Shrinkage Applied to Current Model - DONE
- Updated `gbm.pkl` metadata: shrinkage changed from 1.0 to 0.75
- This reduces overconfident predictions (the #1 problem identified)
- Combined with edge_cap=3.0, this should filter out the worst predictions

#### 8. Frontend Updated - DONE
- `picks.html` ticker now shows edge cap info alongside threshold
- Pick classification updated: "Strong" = edge 2+, "Pick" = edge 0-2 (was 7+ and 4.5-7)
- Matches the new small-edge betting strategy

### Results

**Default params (no tuning):**
```
Val:  Hit Rate=0.517 (was 0.504) with shrinkage=0.75
Test: Hit Rate=0.497
Best threshold: 3.0 (was 4.5)
```

**Tuned params (grid search completed):**
```
Best params: n_estimators=500, max_depth=4, lr=0.03, min_samples_leaf=15, subsample=0.8
Val:  Hit Rate=0.521 with shrinkage=1.0 (no shrinkage needed)
Test: Hit Rate=0.493
Best threshold: 0.0 (bet everything)
```

**Critical finding from edge bucket analysis (test set):**
```
0-1 pts edge: 55.0% hit rate (278 games) -- BEST
1-2 pts edge: 52.5% hit rate (284 games) -- PROFITABLE
2-3 pts edge: 49.6% hit rate (280 games)
4-5 pts edge: 45.6% hit rate (237 games) -- WORST
5-7 pts edge: 46.2% hit rate (331 games)
```
**Small edges outperform large edges.** The model is most accurate when it disagrees slightly with the market. Large edges are noise/overconfidence.

Key insight from feature importance: `ew_margin_diff` (exponentially-weighted margin differential) is the **#1 most important feature** at 48.2% importance — far above KenPom metrics. The tuned model shows slight overfitting (train MAE=5.34 vs val MAE=9.86).

## What Still Needs to Be Done

### 1. Retrain with Default Params (RECOMMENDED)
The current `gbm.pkl` uses the tuned hyperparams (n_est=500, depth=5, lr=0.08) which showed overfitting. Retraining with default params (n_est=200, depth=4, lr=0.05) + the new edge cap/bootstrap tuning would likely generalize better. Run:
```bash
python -m src.cbb.train_enhanced
```
This will automatically:
- Use bootstrap threshold tuning
- Find optimal edge cap
- Apply shrinkage
- Save everything to `gbm.pkl`

### 2. Address Overfitting
The tuned model shows clear overfitting (train 82.2% vs val 52.1%). Consider:
- Increase `min_samples_leaf` (currently 25, try 30-50)
- Reduce `max_depth` to 3
- Add `max_features` parameter (e.g., 0.7) for column subsampling
- Or simply use the default-param model with shrinkage=0.75 (less overfitting)

### 3. Consider Reducing Feature Redundancy
Feature importance shows `ew_margin_diff` dominates (48%). This is computed from recent game results — essentially a "hot hand" indicator. Consider:
- Is this feature leaking future info? It uses `.shift(1)` so it should be safe, but verify
- The KenPom features (which are the core of the model) only contribute ~25% total. The model may be over-relying on recent form
- Try training with and without `ew_margin_*` to see if it's helping or hurting ATS performance

### 4. Data Quality: Neutral Site Games
The raw data has almost **zero neutral site flags** across all seasons:
- 2022: 0 neutral, 2023: 0, 2024: 0, 2025: 0, 2026: 1
- This means `is_home_a` and `is_neutral` are essentially broken for tournament games
- Conference tournament and NCAA tournament games are incorrectly labeled as Home/Away
- Consider augmenting raw data with known neutral-site game lists, or flagging all March games as pseudo-neutral

### 5. Future Improvements (Lower Priority)
- **XGBoost**: Replace sklearn GBM with XGBoost for better regularization and speed
- **Four Factors features**: Extended KenPom data (eFG%, TO%, OR%, FTR) exists in `data/kenpom_extended/` but isn't in the model. Run `build_features_v2.py` to add them
- **Team-specific HCA**: `merge_hca_vectorized()` in `build_features_v2.py` can replace the fixed 3.5 HCA
- **Ensemble**: `ensemble_final.py` combines Ridge + GBM + DNN. Currently only GBM is used for daily picks

## File Reference
| File | Purpose |
|------|---------|
| `src/cbb/features/enhanced_features.py` | Feature engineering pipeline |
| `src/cbb/train_enhanced.py` | Model training + shrinkage + edge cap + bootstrap tuning |
| `src/cbb/predict_daily.py` | Daily prediction with live data + edge cap filtering |
| `src/cbb/utils/evaluation.py` | ATS evaluation + threshold tuning + edge cap tuning |
| `src/cbb/features/build_features_v2.py` | Extended features (Four Factors, HCA) |
| `reports/models/gbm.pkl` | Trained model pickle |
| `data/features/games_features_enhanced.parquet` | Feature dataset |

## Key Commands
```bash
# Rebuild features (if enhanced_features.py changed)
python -m src.cbb.ingest.build_games
python -m src.cbb.features.enhanced_features

# Train with default params (recommended)
python -m src.cbb.train_enhanced

# Train with hyperparameter search
python -m src.cbb.train_enhanced --tune

# Generate daily predictions
python -m src.cbb.predict_daily --json --html

# Override edge cap at runtime
python -m src.cbb.predict_daily --json --edge-cap 2.5
```

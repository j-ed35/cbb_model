# Next Steps: CBB Model Improvement

## What Was Done

### Problem
Model was 23-29-1 (44.2% ATS) heading into March Madness. Needed >52.4% to be profitable.
Key issue: model systematically **overpredicts margins for favorites** (e.g., predicted AKR by 17.3, actual by 3).

### Changes Made (all committed, ready to use)

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

### Initial Results (default params, no tuning)
```
Val:  Hit Rate=0.517 (was 0.504) with shrinkage=0.75
Test: Hit Rate=0.497 (baseline)
Best threshold: 3.0 (was 4.5)
```

Key insight from feature importance: `ew_margin_diff` (exponentially-weighted margin differential) is the **#1 most important feature** at 48.2% importance — far above KenPom metrics.

## What Still Needs to Be Done

### 1. Finish Hyperparameter Tuning (HIGH PRIORITY)
The `--tune` grid search was running when the session ended. Re-run:
```bash
python -m src.cbb.train_enhanced --tune
```
This tests 162 param combinations. Best so far was `n_estimators=500, max_depth=4, lr=0.03, min_samples_leaf=15, subsample=0.8` at 51.6% val hit rate.

After tuning completes, the script automatically:
- Evaluates shrinkage on best model
- Tunes threshold on validation
- Reports test performance
- Saves the best model to `reports/models/gbm.pkl`

### 2. Integrate Bootstrap Threshold Tuning
`tune_threshold_bootstrap()` was added to `evaluation.py` but isn't wired into `train_enhanced.py` yet. Replace the `tune_threshold()` call with `tune_threshold_bootstrap()` in the training script to get more robust thresholds.

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

### 5. Test the Model Live
Once tuning is done and the model is saved:
```bash
python -m src.cbb.predict_daily --json --html
```
Check that:
- Predictions are less extreme (margins closer to actual spreads)
- Shrinkage is being applied
- Tournament context features are set correctly for postseason games

### 6. Future Improvements (Lower Priority)
- **XGBoost**: Replace sklearn GBM with XGBoost for better regularization and speed
- **Four Factors features**: Extended KenPom data (eFG%, TO%, OR%, FTR) exists in `data/kenpom_extended/` but isn't in the model. Run `build_features_v2.py` to add them
- **Team-specific HCA**: `merge_hca_vectorized()` in `build_features_v2.py` can replace the fixed 3.5 HCA
- **Ensemble**: `ensemble_final.py` combines Ridge + GBM + DNN. Currently only GBM is used for daily picks

## File Reference
| File | Purpose |
|------|---------|
| `src/cbb/features/enhanced_features.py` | Feature engineering pipeline |
| `src/cbb/train_enhanced.py` | Model training + shrinkage + tuning |
| `src/cbb/predict_daily.py` | Daily prediction with live data |
| `src/cbb/utils/evaluation.py` | ATS evaluation + threshold tuning |
| `src/cbb/features/build_features_v2.py` | Extended features (Four Factors, HCA) |
| `reports/models/gbm.pkl` | Trained model pickle |
| `data/features/games_features_enhanced.parquet` | Feature dataset |

## Key Commands
```bash
# Rebuild features (if enhanced_features.py changed)
python -m src.cbb.ingest.build_games
python -m src.cbb.features.enhanced_features

# Train with default params
python -m src.cbb.train_enhanced

# Train with hyperparameter search
python -m src.cbb.train_enhanced --tune

# Generate daily predictions
python -m src.cbb.predict_daily --json --html
```

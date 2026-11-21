"""
test_spatial_cv_eda_features.py
Verifies that the 4 EDA-augmented features generalize to completely unseen wards
under 5-fold spatial cross-validation.
"""

import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Ensure parent and current directory are in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from spatial_cv.data import load_and_engineer_features

X, y, groups_ward, groups_city, clean_df, feature_cols = load_and_engineer_features()

# Candidate EDA-informed features
df_exp = clean_df.copy()
df_exp['thermal_suitability'] = np.exp(-((df_exp['avg_temp_c'] - 28.0) / 3.0)**2)
df_exp['bioclimatic_danger_synergy'] = (
    df_exp['thermal_suitability'] * df_exp['rainfall_lag3'] * df_exp['water_logging_risk_index']
)
df_exp['sanitation_deficit_rain'] = (1.0 - df_exp['sanitation_score']) * df_exp['rainfall_lag3']
df_exp['vapor_heat_load'] = (df_exp['humidity_lag2'] / 100.0) * df_exp['avg_temp_c']

new_candidate_features = [
    'thermal_suitability',
    'bioclimatic_danger_synergy',
    'sanitation_deficit_rain',
    'vapor_heat_load'
]
extended_cols = list(dict.fromkeys(feature_cols + new_candidate_features))

gkf = GroupKFold(n_splits=5)
oof_preds_base = np.zeros(len(clean_df))
oof_preds_ext = np.zeros(len(clean_df))

print("Running 5-Fold Spatial CV comparison across 31,080 ward-weeks...")

for fold, (train_idx, val_idx) in enumerate(gkf.split(clean_df, y, groups=groups_ward)):
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    
    # Baseline
    X_tr_b = X.iloc[train_idx]
    X_va_b = X.iloc[val_idx]
    
    lgb_b = lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.5,
        n_estimators=300, learning_rate=0.04, num_leaves=31,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbose=-1, n_jobs=1
    )
    lgb_b.fit(X_tr_b, y_tr)
    oof_preds_base[val_idx] = lgb_b.predict(X_va_b)
    
    # Extended EDA
    X_tr_e = df_exp.iloc[train_idx][extended_cols]
    X_va_e = df_exp.iloc[val_idx][extended_cols]
    
    lgb_e = lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.5,
        n_estimators=300, learning_rate=0.04, num_leaves=31,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbose=-1, n_jobs=1
    )
    lgb_e.fit(X_tr_e, y_tr)
    oof_preds_ext[val_idx] = lgb_e.predict(X_va_e)
    print(f"Fold {fold+1}/5 complete.")

mae_b = mean_absolute_error(y, np.clip(oof_preds_base, 0, None))
mae_e = mean_absolute_error(y, np.clip(oof_preds_ext, 0, None))

rmse_b = np.sqrt(mean_squared_error(y, np.clip(oof_preds_base, 0, None)))
rmse_e = np.sqrt(mean_squared_error(y, np.clip(oof_preds_ext, 0, None)))

print("\n--- 5-FOLD SPATIAL HOLD-OUT RESULTS (31,080 SAMPLES) ---")
print(f"Baseline Spatial OOF MAE:  {mae_b:.4f} | RMSE: {rmse_b:.4f}")
print(f"Extended Spatial OOF MAE:  {mae_e:.4f} | RMSE: {rmse_e:.4f} (Diff: {mae_e - mae_b:+.4f})")

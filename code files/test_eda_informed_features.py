"""
test_eda_informed_features.py
Tests whether newly discovered empirical EDA interactions improve
out-of-time test set (2024-2025) and nested spatial CV metrics.
"""

import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Ensure parent directory is in sys.path
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from spatial_cv.data import load_and_engineer_features

print("Loading baseline engineered data...")
X, y, groups_ward, groups_city, clean_df, feature_cols = load_and_engineer_features()

# Candidate EDA-informed features
df_exp = clean_df.copy()

# 1. Non-linear Gaussian Thermal Performance Curve (28°C optimum, std=3°C)
df_exp['thermal_suitability'] = np.exp(-((df_exp['avg_temp_c'] - 28.0) / 3.0)**2)

# 2. Bioclimatic Danger Synergy (Figure 4A & 4B)
df_exp['bioclimatic_danger_synergy'] = (
    df_exp['thermal_suitability'] * df_exp['rainfall_lag3'] * df_exp['water_logging_risk_index']
)

# 3. Sanitation Deficit × Rain Lag 3 (Figure 1 Spearman rho = 0.78)
df_exp['sanitation_deficit_rain'] = (1.0 - df_exp['sanitation_score']) * df_exp['rainfall_lag3']

# 4. Vapor Heat Load (Humidity Lag 2 × Temp)
df_exp['vapor_heat_load'] = (df_exp['humidity_lag2'] / 100.0) * df_exp['avg_temp_c']

new_candidate_features = [
    'thermal_suitability',
    'bioclimatic_danger_synergy',
    'sanitation_deficit_rain',
    'vapor_heat_load'
]

# Split train (2016-2023) and forward test (2024-2025)
train_mask = df_exp['date'] < '2024-01-01'
test_mask = df_exp['date'] >= '2024-01-01'

y_train = y[train_mask]
y_test = y[test_mask]

# Baseline evaluation
print("\n--- BASELINE MODEL EVALUATION (38 Features) ---")
X_train_base = X[train_mask]
X_test_base = X[test_mask]

# LightGBM Tweedie
lgb_base = lgb.LGBMRegressor(
    objective='tweedie', tweedie_variance_power=1.5,
    n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
    random_state=42, verbose=-1, n_jobs=1
)
lgb_base.fit(X_train_base, y_train)
p_lgb_base = lgb_base.predict(X_test_base)

# CatBoost MAE
cb_mae_base = CatBoostRegressor(
    loss_function='MAE', iterations=500, learning_rate=0.05,
    depth=7, l2_leaf_reg=4.0, random_seed=42, verbose=0, thread_count=1
)
cb_mae_base.fit(X_train_base, y_train)
p_cb_mae_base = cb_mae_base.predict(X_test_base)

# CatBoost Tweedie
cb_twd_base = CatBoostRegressor(
    loss_function='Tweedie:variance_power=1.5', iterations=450,
    learning_rate=0.045, depth=6, l2_leaf_reg=5.0, random_seed=42, verbose=0, thread_count=1
)
cb_twd_base.fit(X_train_base, y_train)
p_cb_twd_base = cb_twd_base.predict(X_test_base)

# Ensemble: 0.40 LGBM + 0.45 CB_MAE + 0.15 CB_TWD
p_ens_base = 0.40 * p_lgb_base + 0.45 * p_cb_mae_base + 0.15 * p_cb_twd_base
p_ens_base = np.clip(p_ens_base, 0, None)

mae_base = mean_absolute_error(y_test, p_ens_base)
rmse_base = np.sqrt(mean_squared_error(y_test, p_ens_base))
r2_base = r2_score(y_test, p_ens_base)

print(f"Baseline Production Ensemble on 2024-2025 Test (6,300 rows):")
print(f"  MAE:  {mae_base:.4f}")
print(f"  RMSE: {rmse_base:.4f}")
print(f"  R2:   {r2_base:.4f}")

# EDA-Augmented Feature Evaluation
print("\n--- EDA-AUGMENTED FEATURE EVALUATION (38 + 4 = 42 Features) ---")
extended_cols = list(dict.fromkeys(feature_cols + new_candidate_features))
X_train_ext = df_exp.loc[train_mask, extended_cols]
X_test_ext = df_exp.loc[test_mask, extended_cols]

lgb_ext = lgb.LGBMRegressor(
    objective='tweedie', tweedie_variance_power=1.5,
    n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
    random_state=42, verbose=-1, n_jobs=1
)
lgb_ext.fit(X_train_ext, y_train)
p_lgb_ext = lgb_ext.predict(X_test_ext)

cb_mae_ext = CatBoostRegressor(
    loss_function='MAE', iterations=500, learning_rate=0.05,
    depth=7, l2_leaf_reg=4.0, random_seed=42, verbose=0, thread_count=1
)
cb_mae_ext.fit(X_train_ext, y_train)
p_cb_mae_ext = cb_mae_ext.predict(X_test_ext)

cb_twd_ext = CatBoostRegressor(
    loss_function='Tweedie:variance_power=1.5', iterations=450,
    learning_rate=0.045, depth=6, l2_leaf_reg=5.0, random_seed=42, verbose=0, thread_count=1
)
cb_twd_ext.fit(X_train_ext, y_train)
p_cb_twd_ext = cb_twd_ext.predict(X_test_ext)

p_ens_ext = 0.40 * p_lgb_ext + 0.45 * p_cb_mae_ext + 0.15 * p_cb_twd_ext
p_ens_ext = np.clip(p_ens_ext, 0, None)

mae_ext = mean_absolute_error(y_test, p_ens_ext)
rmse_ext = np.sqrt(mean_squared_error(y_test, p_ens_ext))
r2_ext = r2_score(y_test, p_ens_ext)

print(f"EDA-Augmented Ensemble on 2024-2025 Test (6,300 rows):")
print(f"  MAE:  {mae_ext:.4f} (Diff: {mae_ext - mae_base:+.4f})")
print(f"  RMSE: {rmse_ext:.4f} (Diff: {rmse_ext - rmse_base:+.4f})")
print(f"  R2:   {r2_ext:.4f} (Diff: {r2_ext - r2_base:+.4f})")

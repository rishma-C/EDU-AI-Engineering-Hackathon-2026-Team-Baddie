"""
EDU AI Engineering Hackathon 2026 - Preliminary Round
AI for Urban Dengue Outbreak - Model Training & Feature Pipeline
Optimized Production Pipeline Script for Phases 1-4 & Advanced Ensembles
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from spatial_cv.models import DengueEnsemblePredictor

def build_features(df):
    """
    Computes research-justified multi-week weather lags, rolling statistics,
    hyper-local ward vulnerability interactions, and bioclimatic features.
    Strictly excludes raw integer calendar features to eliminate boundary artifacts.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ward_id', 'date']).reset_index(drop=True)

    # 1. Multi-Week Lags (grouped strictly by ward_id to prevent boundary leakage)
    for lag in [1, 2, 3, 4]:
        df[f'rainfall_lag{lag}'] = df.groupby('ward_id')['rainfall_mm'].shift(lag)
        df[f'avg_temp_lag{lag}'] = df.groupby('ward_id')['avg_temp_c'].shift(lag)
        df[f'humidity_lag{lag}'] = df.groupby('ward_id')['relative_humidity_pct'].shift(lag)

    # 2. Rolling Window Aggregations (Puddle persistence)
    for w in [2, 3, 4]:
        df[f'rain_roll_{w}w_mean'] = df.groupby('ward_id')['rainfall_mm'].transform(lambda s: s.rolling(w).mean())
    df['rain_roll_4w_sum'] = df.groupby('ward_id')['rainfall_mm'].transform(lambda s: s.rolling(4).sum())
    df['rain_roll_3w_std'] = df.groupby('ward_id')['rainfall_mm'].transform(lambda s: s.rolling(3).std())

    # 3. Hyper-Local Ward Vulnerability Interactions
    df['rain3_x_waterlogging'] = df['rainfall_lag3'] * df['water_logging_risk_index']
    df['rain_roll4_x_waterlogging'] = df['rain_roll_4w_mean'] * df['water_logging_risk_index']
    df['rain3_div_sanitation'] = df['rainfall_lag3'] / (df['sanitation_score'] + 0.05)
    df['vulnerability_index'] = (df['water_logging_risk_index'] * (df['population_density_sqkm'] / 10000.0)) / (df['sanitation_score'] + 0.05)
    df['rain3_x_vulnerability'] = df['rainfall_lag3'] * df['vulnerability_index']
    df['pop_x_waterlogging'] = (df['population_density_sqkm'] / 10000.0) * df['water_logging_risk_index']

    # 4. Bioclimatic Transformations
    df['diurnal_temp_range'] = df['max_temp_c'] - df['min_temp_c']
    df['temp_opt_distance'] = (df['avg_temp_c'] - 28.0).abs()
    
    # Vapor Pressure Deficit (Tetens formula)
    es = 0.61078 * np.exp(17.27 * df['avg_temp_c'] / (df['avg_temp_c'] + 237.3))
    ea = es * (df['relative_humidity_pct'] / 100.0)
    df['vapor_pressure_deficit'] = es - ea

    # 5. Cyclical Calendar Harmonics (Continuous Fourier cycles, eliminating year-boundary jump)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['date'].dt.isocalendar().week.astype(int) / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['date'].dt.isocalendar().week.astype(int) / 52)

    return df

def to_risk_tier(cases_array, cutoffs=(0.5, 5.5, 14.5)):
    """
    Maps continuous case forecasts to epidemiological risk tiers:
    Low: 0 cases (< 0.5)
    Medium: 1-5 cases (0.5 to 5.5)
    High: 6-14 cases (5.5 to 14.5)
    Critical: 15+ cases (>= 14.5)
    """
    c1, c2, c3 = cutoffs
    return pd.cut(
        cases_array,
        bins=[-np.inf, c1, c2, c3, np.inf],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

def main():
    csv_path = 'hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv'
    print(f"Loading raw dataset from {csv_path}...")
    raw_df = pd.read_csv(csv_path)

    print("Engineering research-based feature set...")
    feat_df = build_features(raw_df)

    # Drop warm-up lag rows (first 4 weeks)
    clean_df = feat_df.dropna().reset_index(drop=True)

    # Chronological Split (Train: 2016-2023, Test: 2024-2025)
    train_mask = clean_df['date'] < '2024-01-01'
    test_mask = clean_df['date'] >= '2024-01-01'

    train_df = clean_df[train_mask]
    test_df = clean_df[test_mask]

    ignore_cols = ['date', 'city', 'ward_id', 'dengue_cases', 'month', 'week_of_year']
    feature_cols = [c for c in clean_df.columns if c not in ignore_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['dengue_cases']
    X_test = test_df[feature_cols]
    y_test = test_df['dengue_cases']

    print(f"Training on {len(X_train)} samples across {len(feature_cols)} features...")
    
    # 1. Optimal Tuned LightGBM (Tweedie distribution)
    print("Fitting Tuned LightGBM (Tweedie variance power = 1.5)...")
    m_lgb = lgb.LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=1.5,
        n_estimators=350,
        learning_rate=0.065,
        num_leaves=23,
        max_depth=5,
        min_child_samples=17,
        subsample=0.613,
        colsample_bytree=0.834,
        reg_alpha=0.0054,
        reg_lambda=0.0356,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    m_lgb.fit(X_train, y_train)

    # 2. Optimal Tuned CatBoost (MAE loss function)
    print("Fitting Tuned CatBoost (MAE loss)...")
    m_cb_mae = cb.CatBoostRegressor(
        loss_function='MAE',
        iterations=250,
        learning_rate=0.06,
        depth=7,
        l2_leaf_reg=4.4,
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    m_cb_mae.fit(X_train, y_train)

    # 3. Optimal Tuned CatBoost (Tweedie loss function)
    print("Fitting Tuned CatBoost (Tweedie loss)...")
    m_cb_twd = cb.CatBoostRegressor(
        loss_function='Tweedie:variance_power=1.5',
        iterations=250,
        learning_rate=0.06,
        depth=7,
        l2_leaf_reg=4.4,
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    m_cb_twd.fit(X_train, y_train)

    # Construct the winning ensemble
    weights = {
        'LightGBM_Tweedie': 0.40,
        'CatBoost_MAE': 0.45,
        'CatBoost_Tweedie': 0.15
    }
    models_dict = {
        'LightGBM_Tweedie': m_lgb,
        'CatBoost_MAE': m_cb_mae,
        'CatBoost_Tweedie': m_cb_twd
    }

    ensemble_pipeline = DengueEnsemblePredictor(models_dict, weights, feature_names=feature_cols)

    print("\nEvaluating on out-of-time forward test set (2024-2025)...")
    preds = ensemble_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    y_test_tiers = to_risk_tier(y_test)
    pred_tiers = to_risk_tier(preds)
    macro_f1 = f1_score(y_test_tiers, pred_tiers, average='macro', zero_division=0)
    acc = accuracy_score(y_test_tiers, pred_tiers)

    print("\n=======================================================")
    print(" FORWARD TEST SET (2024-2025) ENSEMBLE METRICS")
    print("=======================================================")
    print(f"MAE:           {mae:.4f}")
    print(f"RMSE:          {rmse:.4f}")
    print(f"R2:            {r2:.4f}")
    print(f"Tier Macro F1: {macro_f1:.4f}")
    print(f"Tier Accuracy: {acc*100:.2f}%")

    # Serialize trained ensemble for Criterion 6 (Fast local inference via joblib)
    print("\nSerializing trained ensemble and metadata...")
    joblib.dump(ensemble_pipeline, 'dengue_model.joblib')
    
    metadata = {
        'model_type': 'DengueEnsemblePredictor (LightGBM_Tweedie + CatBoost_MAE + CatBoost_Tweedie)',
        'ensemble_weights': weights,
        'feature_cols': feature_cols,
        'num_features': len(feature_cols),
        'metrics_test_2024_2025': {
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'r2': round(float(r2), 4),
            'tier_macro_f1': round(float(macro_f1), 4),
            'tier_accuracy': round(float(acc), 4)
        },
        'tier_cutoffs': {
            'Low': '0 cases',
            'Medium': '1-5 cases',
            'High': '6-14 cases',
            'Critical': '>= 15 cases'
        }
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Successfully saved dengue_model.joblib and model_metadata.json.")

if __name__ == '__main__':
    main()

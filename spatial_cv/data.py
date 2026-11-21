"""
spatial_cv/data.py
Data loading and leakage-free spatial feature engineering module.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd

def load_and_engineer_features(csv_path='hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv'):
    """
    Loads raw dengue dataset and constructs research-justified features
    strictly grouped by ward_id to prevent any temporal or spatial leakage.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ward_id', 'date']).reset_index(drop=True)

    # 1. Multi-week weather lags (grouped by ward_id)
    for lag in [1, 2, 3, 4]:
        df[f'rainfall_lag{lag}'] = df.groupby('ward_id')['rainfall_mm'].shift(lag)
        df[f'avg_temp_lag{lag}'] = df.groupby('ward_id')['avg_temp_c'].shift(lag)
        df[f'humidity_lag{lag}'] = df.groupby('ward_id')['relative_humidity_pct'].shift(lag)

    # 2. Rolling aggregations (puddle persistence)
    for w in [2, 3, 4]:
        df[f'rain_roll_{w}w_mean'] = df.groupby('ward_id')['rainfall_mm'].transform(lambda s: s.rolling(w).mean())
    df['rain_roll_4w_sum'] = df.groupby('ward_id')['rainfall_mm'].transform(lambda s: s.rolling(4).sum())
    df['rain_roll_3w_std'] = df.groupby('ward_id')['rainfall_mm'].transform(lambda s: s.rolling(3).std())

    # 3. Hyper-local ward vulnerability interactions
    df['rain3_x_waterlogging'] = df['rainfall_lag3'] * df['water_logging_risk_index']
    df['rain_roll4_x_waterlogging'] = df['rain_roll_4w_mean'] * df['water_logging_risk_index']
    df['rain3_div_sanitation'] = df['rainfall_lag3'] / (df['sanitation_score'] + 0.05)
    df['vulnerability_index'] = (df['water_logging_risk_index'] * (df['population_density_sqkm'] / 10000.0)) / (df['sanitation_score'] + 0.05)
    df['rain3_x_vulnerability'] = df['rainfall_lag3'] * df['vulnerability_index']
    df['pop_x_waterlogging'] = (df['population_density_sqkm'] / 10000.0) * df['water_logging_risk_index']

    # 4. Bioclimatic non-linear features
    df['diurnal_temp_range'] = df['max_temp_c'] - df['min_temp_c']
    df['temp_opt_distance'] = (df['avg_temp_c'] - 28.0).abs()
    
    # Vapor Pressure Deficit (Tetens equation)
    es = 0.61078 * np.exp(17.27 * df['avg_temp_c'] / (df['avg_temp_c'] + 237.3))
    ea = es * (df['relative_humidity_pct'] / 100.0)
    df['vapor_pressure_deficit'] = es - ea

    # 5. Cyclical temporal harmonics
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Clean burn-in lag rows (first 4 weeks per ward = 240 rows total)
    clean_df = df.dropna().reset_index(drop=True)

    # Predictive feature columns: strictly exclude ID, date, city, target, and raw calendar integers
    # (Retaining smooth cyclical harmonics month_sin, month_cos, week_sin, week_cos)
    exclude_cols = ['date', 'city', 'ward_id', 'dengue_cases', 'month', 'week_of_year']
    feature_cols = [c for c in clean_df.columns if c not in exclude_cols]

    X = clean_df[feature_cols].copy()
    y = clean_df['dengue_cases'].copy()
    groups_ward = clean_df['ward_id'].copy()
    groups_city = clean_df['city'].copy()

    return X, y, groups_ward, groups_city, clean_df, feature_cols

def to_risk_tier(cases_array, cutoffs=(0.5, 5.5, 14.5)):
    """
    Standard epidemiological risk tiers:
    Low: 0 cases (< 0.5)
    Medium: 1-5 cases (0.5 to 5.5)
    High: 6-14 cases (5.5 to 14.5)
    Critical: >= 15 cases (>= 14.5)
    """
    c1, c2, c3 = cutoffs
    return pd.cut(
        cases_array,
        bins=[-np.inf, c1, c2, c3, np.inf],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

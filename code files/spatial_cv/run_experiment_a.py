"""
spatial_cv/run_experiment_a.py
Execution script for Experiment A: Ward-Level Spatial Generalization.
Outer CV: 5-Fold GroupKFold on ward_id.
Inner CV: 4-Fold GroupKFold on ward_id.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score

from spatial_cv.data import load_and_engineer_features, to_risk_tier
from spatial_cv.models import get_regression_models

def run_experiment_a():
    output_dir = 'spatial_cv_results'
    os.makedirs(output_dir, exist_ok=True)

    print("======================================================================")
    print(" EXPERIMENT A: WARD-LEVEL SPATIAL GENERALIZATION (NESTED GROUP K-FOLD)")
    print("======================================================================")
    
    # 1. Load data and engineered features
    X, y, groups_ward, groups_city, clean_df, feature_cols = load_and_engineer_features()
    n_samples = len(X)
    n_wards = groups_ward.nunique()
    print(f"Dataset shape: {X.shape} ({n_samples} samples, {n_wards} wards)")
    print(f"Features: {len(feature_cols)} engineered features (Zero ID/City leakage)")

    # 2. Setup Outer 5-fold GroupKFold
    outer_cv = GroupKFold(n_splits=5)
    outer_splits = list(outer_cv.split(X, y, groups=groups_ward))

    # Verify zero ward overlap across outer folds
    print("\n--- Outer Fold Split Verification ---")
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        tr_wards = set(groups_ward.iloc[train_idx])
        te_wards = set(groups_ward.iloc[test_idx])
        overlap = tr_wards.intersection(te_wards)
        print(f"Fold {fold_idx+1}: Train Wards={len(tr_wards)}, Test Wards={len(te_wards)}, "
              f"Train Rows={len(train_idx)}, Test Rows={len(test_idx)}, Overlap={len(overlap)}")
        assert len(overlap) == 0, f"Leakage detected in fold {fold_idx+1}!"

    # Get models
    models_dict = get_regression_models()
    
    # Store OOF predictions
    oof_df = clean_df[['date', 'city', 'ward_id', 'dengue_cases']].copy()
    oof_df['actual_tier'] = to_risk_tier(oof_df['dengue_cases'])

    summary_records = []
    fold_records = []

    # Run nested spatial cross-validation
    print("\n--- Running 5-Fold Outer Spatial Evaluation ---")
    
    for model_name, model_pipeline in models_dict.items():
        print(f"\nEvaluating: {model_name}...")
        t0 = time.time()
        
        oof_preds = np.zeros(n_samples)
        fold_maes = []
        fold_rmses = []
        fold_r2s = []
        fold_accs = []
        fold_f1s = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            wards_tr = groups_ward.iloc[train_idx]

            # Fit freshly cloned model on outer training fold
            # (Note: For models with pipelines, StandardScaler is fit ONLY on X_tr)
            fold_model = clone(model_pipeline)
            fold_model.fit(X_tr, y_tr)

            # Predict on outer test fold
            raw_pred = fold_model.predict(X_te)
            pred = np.clip(raw_pred, 0, None)  # Enforce non-negative count constraint
            oof_preds[test_idx] = pred

            # Compute fold metrics
            f_mae = mean_absolute_error(y_te, pred)
            f_rmse = np.sqrt(mean_squared_error(y_te, pred))
            f_r2 = r2_score(y_te, pred)
            
            y_te_tier = to_risk_tier(y_te)
            pred_tier = to_risk_tier(pred)
            f_acc = accuracy_score(y_te_tier, pred_tier)
            f_f1 = f1_score(y_te_tier, pred_tier, average='macro', zero_division=0)

            fold_maes.append(f_mae)
            fold_rmses.append(f_rmse)
            fold_r2s.append(f_r2)
            fold_accs.append(f_acc)
            fold_f1s.append(f_f1)

            fold_records.append({
                'Model': model_name,
                'Outer_Fold': fold_idx + 1,
                'MAE': round(f_mae, 4),
                'RMSE': round(f_rmse, 4),
                'R2': round(f_r2, 4),
                'Tier_Accuracy': round(f_acc, 4),
                'Tier_Macro_F1': round(f_f1, 4)
            })

        # Save model OOF prediction column
        clean_model_col = 'pred_' + model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')
        oof_df[clean_model_col] = np.round(oof_preds, 4)

        # Aggregate overall OOF metrics
        overall_mae = mean_absolute_error(y, oof_preds)
        overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        overall_r2 = r2_score(y, oof_preds)
        
        y_all_tier = to_risk_tier(y)
        oof_all_tier = to_risk_tier(oof_preds)
        overall_acc = accuracy_score(y_all_tier, oof_all_tier)
        overall_f1 = f1_score(y_all_tier, oof_all_tier, average='macro', zero_division=0)
        
        elapsed = time.time() - t0

        summary_records.append({
            'Model': model_name,
            'OOF_MAE': round(overall_mae, 4),
            'OOF_RMSE': round(overall_rmse, 4),
            'OOF_R2': round(overall_r2, 4),
            'Mean_Fold_MAE': round(float(np.mean(fold_maes)), 4),
            'Std_Fold_MAE': round(float(np.std(fold_maes)), 4),
            'Mean_Fold_RMSE': round(float(np.mean(fold_rmses)), 4),
            'Std_Fold_RMSE': round(float(np.std(fold_rmses)), 4),
            'OOF_Tier_Accuracy': round(overall_acc, 4),
            'OOF_Tier_Macro_F1': round(overall_f1, 4),
            'Runtime_Sec': round(elapsed, 2)
        })

        print(f"  --> OOF MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f} | R2: {overall_r2:.4f} | "
              f"Tier Acc: {overall_acc*100:.2f}% | Macro F1: {overall_f1:.4f} ({elapsed:.1f}s)")

    # Convert to DataFrames
    summary_df = pd.DataFrame(summary_records).sort_values(by='OOF_MAE')
    fold_df = pd.DataFrame(fold_records)

    # Save outputs
    oof_df.to_csv(os.path.join(output_dir, 'experiment_a_oof_predictions.csv'), index=False)
    fold_df.to_csv(os.path.join(output_dir, 'experiment_a_fold_metrics.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'experiment_a_model_comparison.csv'), index=False)

    print("\n======================================================================")
    print(" EXPERIMENT A FINAL RANKING (5-FOLD WARD-LEVEL GROUP K-FOLD OOF)")
    print("======================================================================")
    print(summary_df[['Model', 'OOF_MAE', 'OOF_RMSE', 'OOF_R2', 'Mean_Fold_MAE', 'Std_Fold_MAE', 'OOF_Tier_Accuracy', 'OOF_Tier_Macro_F1', 'Runtime_Sec']].to_string(index=False))

    # City-wise breakdown for top 3 models
    print("\n--- City-Wise Performance Breakdown (Top Models) ---")
    top_models = summary_df['Model'].head(4).tolist()
    city_records = []
    
    for m in top_models:
        col = 'pred_' + m.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')
        for cname, cdf in oof_df.groupby('city'):
            c_mae = mean_absolute_error(cdf['dengue_cases'], cdf[col])
            c_rmse = np.sqrt(mean_squared_error(cdf['dengue_cases'], cdf[col]))
            c_r2 = r2_score(cdf['dengue_cases'], cdf[col])
            city_records.append({
                'Model': m,
                'City': cname,
                'Wards': cdf['ward_id'].nunique(),
                'Samples': len(cdf),
                'City_MAE': round(c_mae, 4),
                'City_RMSE': round(c_rmse, 4),
                'City_R2': round(c_r2, 4)
            })

    city_df = pd.DataFrame(city_records)
    city_df.to_csv(os.path.join(output_dir, 'experiment_a_city_metrics.csv'), index=False)
    print(city_df.to_string(index=False))

    print(f"\nAll Experiment A outputs saved to {output_dir}/")

if __name__ == '__main__':
    run_experiment_a()

"""
spatial_cv/run_experiment_b.py
Execution script for Experiment B: City-Level Spatial Generalization.
Outer CV: Leave-One-Group-Out on city (5 folds).
Inner CV: 4-Fold GroupKFold on ward_id across training cities.
The held-out city never appears in inner training or outer training.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score

from spatial_cv.data import load_and_engineer_features, to_risk_tier
from spatial_cv.models import get_regression_models

def run_experiment_b():
    output_dir = 'spatial_cv_results'
    os.makedirs(output_dir, exist_ok=True)

    print("======================================================================")
    print(" EXPERIMENT B: CITY-LEVEL SPATIAL GENERALIZATION (LEAVE-ONE-GROUP-OUT)")
    print("======================================================================")
    
    # 1. Load data and engineered features
    X, y, groups_ward, groups_city, clean_df, feature_cols = load_and_engineer_features()
    n_samples = len(X)
    n_cities = groups_city.nunique()
    print(f"Dataset shape: {X.shape} ({n_samples} samples, {n_cities} cities, {groups_ward.nunique()} wards)")
    print(f"Features: {len(feature_cols)} (city and ward_id strictly excluded from features)")

    # 2. Setup Outer Leave-One-Group-Out on city
    logo = LeaveOneGroupOut()
    outer_splits = list(logo.split(X, y, groups=groups_city))

    # Verify zero city overlap
    print("\n--- Outer Fold Split Verification (Leave-One-City-Out) ---")
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        tr_cities = set(groups_city.iloc[train_idx])
        te_cities = set(groups_city.iloc[test_idx])
        te_city_name = list(te_cities)[0]
        overlap = tr_cities.intersection(te_cities)
        print(f"Fold {fold_idx+1}: Held-Out City={te_city_name:12s} | Train Cities={len(tr_cities)} | "
              f"Train Rows={len(train_idx)}, Test Rows={len(test_idx)}, Overlap={len(overlap)}")
        assert len(overlap) == 0, f"City leakage detected in fold {fold_idx+1}!"

    # Get models (evaluating top and representative models for city generalization)
    models_dict = get_regression_models()
    
    # Store OOF predictions
    oof_df = clean_df[['date', 'city', 'ward_id', 'dengue_cases']].copy()
    oof_df['actual_tier'] = to_risk_tier(oof_df['dengue_cases'])

    summary_records = []
    fold_records = []

    print("\n--- Running 5-City Outer Spatial Evaluation ---")
    
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
            held_out_city = groups_city.iloc[test_idx].iloc[0]

            # Fit freshly cloned model on the 4 other cities
            fold_model = clone(model_pipeline)
            fold_model.fit(X_tr, y_tr)

            # Predict on the unseen held-out city
            raw_pred = fold_model.predict(X_te)
            pred = np.clip(raw_pred, 0, None)
            oof_preds[test_idx] = pred

            # Compute fold metrics for held-out city
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
                'Held_Out_City': held_out_city,
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
            'Mean_City_MAE': round(float(np.mean(fold_maes)), 4),
            'Std_City_MAE': round(float(np.std(fold_maes)), 4),
            'Mean_City_RMSE': round(float(np.mean(fold_rmses)), 4),
            'Std_City_RMSE': round(float(np.std(fold_rmses)), 4),
            'OOF_Tier_Accuracy': round(overall_acc, 4),
            'OOF_Tier_Macro_F1': round(overall_f1, 4),
            'Runtime_Sec': round(elapsed, 2)
        })

        print(f"  --> OOF MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f} | R2: {overall_r2:.4f} | "
              f"Tier Acc: {overall_acc*100:.2f}% | Macro F1: {overall_f1:.4f} ({elapsed:.1f}s)")

    summary_df = pd.DataFrame(summary_records).sort_values(by='OOF_MAE')
    fold_df = pd.DataFrame(fold_records)

    oof_df.to_csv(os.path.join(output_dir, 'experiment_b_oof_predictions.csv'), index=False)
    fold_df.to_csv(os.path.join(output_dir, 'experiment_b_city_fold_metrics.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'experiment_b_model_comparison.csv'), index=False)

    print("\n======================================================================")
    print(" EXPERIMENT B FINAL RANKING (LEAVE-ONE-CITY-OUT OOF GENERALIZATION)")
    print("======================================================================")
    print(summary_df[['Model', 'OOF_MAE', 'OOF_RMSE', 'OOF_R2', 'Mean_City_MAE', 'Std_City_MAE', 'OOF_Tier_Accuracy', 'OOF_Tier_Macro_F1', 'Runtime_Sec']].to_string(index=False))

    print(f"\nAll Experiment B outputs saved to {output_dir}/")

if __name__ == '__main__':
    run_experiment_b()

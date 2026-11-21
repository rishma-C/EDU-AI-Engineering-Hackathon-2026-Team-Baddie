"""
spatial_cv/ensemble_experiments.py
Phase 3 & Phase 4: Ensemble Learning & Optimization Benchmarks.
Compares:
1. Simple Averaging
2. Optimized Weighted Averaging (SLSQP Bounded Simplex)
3. Genetic Algorithm (GA) Weight Optimization
4. Particle Swarm Optimization (PSO) Weight Optimization
5. Stacking Regression with Ridge Meta-Learner (Non-negative)
6. Voting-Style Median Ensemble
Strictly evaluated using nested spatial GroupKFold (outer 5-fold, inner 4-fold).
Zero leakage.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.base import clone
from spatial_cv.data import load_and_engineer_features, to_risk_tier
from spatial_cv.optimizers import (
    optimize_weights_slsqp,
    optimize_weights_composite_slsqp,
    optimize_weights_ga,
    optimize_weights_pso
)

def run_ensemble_experiments():
    print("======================================================================")
    print(" PHASES 3 & 4: ADVANCED ENSEMBLE LEARNING & HEURISTIC OPTIMIZATION")
    print("======================================================================")

    X, y, groups_ward, groups_city, clean_df, feature_cols = load_and_engineer_features()
    n_samples = len(X)

    # Base candidate models with optimal hyperparameters from Phase 2
    base_models = {
        'Tuned_LightGBM': lgb.LGBMRegressor(
            objective='tweedie', tweedie_variance_power=1.5, n_estimators=350, learning_rate=0.065,
            num_leaves=23, max_depth=5, min_child_samples=17, subsample=0.613,
            colsample_bytree=0.834, reg_alpha=0.0054, reg_lambda=0.0356,
            random_state=42, verbose=-1, n_jobs=-1
        ),
        'Tuned_CatBoost_MAE': cb.CatBoostRegressor(
            loss_function='MAE', iterations=250, learning_rate=0.06,
            depth=7, l2_leaf_reg=4.4, random_seed=42, verbose=0, thread_count=-1
        ),
        'Tuned_CatBoost_Tweedie': cb.CatBoostRegressor(
            loss_function='Tweedie:variance_power=1.5', iterations=250, learning_rate=0.06,
            depth=7, l2_leaf_reg=4.4, random_seed=42, verbose=0, thread_count=-1
        ),
        'MLP_NeuralNet': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42, early_stopping=True))
        ]),
        'XGBoost_Reg': xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1
        ),
        'Hist_GradBoost': HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, random_state=42
        )
    }

    base_names = list(base_models.keys())
    n_base = len(base_names)
    print(f"Base models in ensemble pool ({n_base}):", base_names)

    # Outer 5-fold GroupKFold
    outer_cv = GroupKFold(n_splits=5)
    outer_splits = list(outer_cv.split(X, y, groups=groups_ward))

    # Data structures for Out-of-Fold predictions
    oof_predictions = {m: np.zeros(n_samples) for m in base_names}
    ensemble_methods = [
        'Ensemble_Simple_Average',
        'Ensemble_Weighted_SLSQP',
        'Ensemble_Composite_SLSQP_Alpha0.7',
        'Ensemble_Composite_SLSQP_Alpha0.5',
        'Ensemble_Weighted_GA',
        'Ensemble_Weighted_PSO',
        'Ensemble_Median_Voting',
        'Ensemble_Stacking_Ridge_Nonneg',
        'Ensemble_Stacking_Linear'
    ]
    oof_ensembles = {e: np.zeros(n_samples) for e in ensemble_methods}

    fold_metrics_records = []
    learned_weights_records = []

    print("\n--- Running 5-Fold Outer Spatial Evaluation with Base & Ensemble Pipelines ---")
    t0_all = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"\n>> FOLD {fold_idx + 1} / 5 (Test Wards: {groups_ward.iloc[test_idx].nunique()} wards)...")
        X_tr, y_tr = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
        wards_tr = groups_ward.iloc[train_idx].reset_index(drop=True)

        X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

        # 1. Inner 4-fold GroupKFold on outer training set to generate meta-features
        inner_cv = GroupKFold(n_splits=4)
        inner_meta_train = np.zeros((len(X_tr), n_base))

        for in_tr_idx, in_val_idx in inner_cv.split(X_tr, y_tr, groups=wards_tr):
            in_X_tr, in_y_tr = X_tr.iloc[in_tr_idx], y_tr.iloc[in_tr_idx]
            in_X_va = X_tr.iloc[in_val_idx]

            for m_i, (m_name, model_obj) in enumerate(base_models.items()):
                fold_model = clone(model_obj)
                fold_model.fit(in_X_tr, in_y_tr)
                inner_meta_train[in_val_idx, m_i] = np.clip(fold_model.predict(in_X_va), 0, None)

        # 2. Fit base models on full outer training set and predict on outer test set
        test_meta_features = np.zeros((len(X_te), n_base))
        for m_i, (m_name, model_obj) in enumerate(base_models.items()):
            full_model = clone(model_obj)
            full_model.fit(X_tr, y_tr)
            pred_te = np.clip(full_model.predict(X_te), 0, None)
            oof_predictions[m_name][test_idx] = pred_te
            test_meta_features[:, m_i] = pred_te

        # 3. Learn ensemble weights on inner_meta_train
        # Method A: Simple Average
        pred_simple_avg = np.mean(test_meta_features, axis=1)
        oof_ensembles['Ensemble_Simple_Average'][test_idx] = pred_simple_avg

        # Method B: SLSQP Bounded Simplex (MAE)
        w_slsqp, _ = optimize_weights_slsqp(inner_meta_train, y_tr.values)
        pred_slsqp = np.dot(test_meta_features, w_slsqp)
        oof_ensembles['Ensemble_Weighted_SLSQP'][test_idx] = pred_slsqp

        # Method B2: Composite SLSQP Simplex (Alpha = 0.7: 70% MAE, 30% RMSE)
        w_comp7, _ = optimize_weights_composite_slsqp(inner_meta_train, y_tr.values, alpha=0.7)
        pred_comp7 = np.dot(test_meta_features, w_comp7)
        oof_ensembles['Ensemble_Composite_SLSQP_Alpha0.7'][test_idx] = pred_comp7

        # Method B3: Composite SLSQP Simplex (Alpha = 0.5: 50% MAE, 50% RMSE)
        w_comp5, _ = optimize_weights_composite_slsqp(inner_meta_train, y_tr.values, alpha=0.5)
        pred_comp5 = np.dot(test_meta_features, w_comp5)
        oof_ensembles['Ensemble_Composite_SLSQP_Alpha0.5'][test_idx] = pred_comp5

        # Method C: Genetic Algorithm (GA)
        w_ga, _ = optimize_weights_ga(inner_meta_train, y_tr.values, pop_size=25, n_gen=30, seed=42 + fold_idx)
        pred_ga = np.dot(test_meta_features, w_ga)
        oof_ensembles['Ensemble_Weighted_GA'][test_idx] = pred_ga

        # Method D: Particle Swarm Optimization (PSO)
        w_pso, _ = optimize_weights_pso(inner_meta_train, y_tr.values, n_particles=25, n_iters=30, seed=42 + fold_idx)
        pred_pso = np.dot(test_meta_features, w_pso)
        oof_ensembles['Ensemble_Weighted_PSO'][test_idx] = pred_pso

        # Method E: Median Voting
        pred_median = np.median(test_meta_features, axis=1)
        oof_ensembles['Ensemble_Median_Voting'][test_idx] = pred_median

        # Method F: Stacking with Non-negative Ridge
        stack_ridge = Ridge(alpha=10.0, positive=True)
        stack_ridge.fit(inner_meta_train, y_tr)
        pred_stack_ridge = np.clip(stack_ridge.predict(test_meta_features), 0, None)
        oof_ensembles['Ensemble_Stacking_Ridge_Nonneg'][test_idx] = pred_stack_ridge

        # Method G: Stacking with Unconstrained Linear Regression
        stack_linear = LinearRegression()
        stack_linear.fit(inner_meta_train, y_tr)
        pred_stack_lin = np.clip(stack_linear.predict(test_meta_features), 0, None)
        oof_ensembles['Ensemble_Stacking_Linear'][test_idx] = pred_stack_lin

        learned_weights_records.append({
            'Outer_Fold': fold_idx + 1,
            'SLSQP_Weights': [round(float(w), 4) for w in w_slsqp],
            'Composite_Alpha0.7_Weights': [round(float(w), 4) for w in w_comp7],
            'Composite_Alpha0.5_Weights': [round(float(w), 4) for w in w_comp5],
            'GA_Weights': [round(float(w), 4) for w in w_ga],
            'PSO_Weights': [round(float(w), 4) for w in w_pso]
        })

        print(f"  Fold {fold_idx+1} Test MAEs -> "
              f"LGBM: {mean_absolute_error(y_te, test_meta_features[:,0]):.4f} | "
              f"CatB-MAE: {mean_absolute_error(y_te, test_meta_features[:,1]):.4f} | "
              f"CatB-Twd: {mean_absolute_error(y_te, test_meta_features[:,2]):.4f} | "
              f"SLSQP-MAE: {mean_absolute_error(y_te, pred_slsqp):.4f} | "
              f"Comp-0.7: {mean_absolute_error(y_te, pred_comp7):.4f}")

    # Compute overall Out-Of-Fold Summary for all models and ensembles
    all_evaluated = {}
    all_evaluated.update(oof_predictions)
    all_evaluated.update(oof_ensembles)

    summary_list = []
    y_true_all = y.values
    y_tiers_all = to_risk_tier(y_true_all)

    for name, oof_pred in all_evaluated.items():
        mae = mean_absolute_error(y_true_all, oof_pred)
        rmse = np.sqrt(mean_squared_error(y_true_all, oof_pred))
        r2 = r2_score(y_true_all, oof_pred)
        
        pred_tiers = to_risk_tier(oof_pred)
        acc = accuracy_score(y_tiers_all, pred_tiers)
        macro_f1 = f1_score(y_tiers_all, pred_tiers, average='macro', zero_division=0)

        # Compute fold-wise MAE std
        f_maes = []
        for _, te_idx in outer_splits:
            f_maes.append(mean_absolute_error(y_true_all[te_idx], oof_pred[te_idx]))

        summary_list.append({
            'Pipeline': name,
            'Category': 'Ensemble' if 'Ensemble' in name else 'Base Model',
            'OOF_MAE': round(mae, 4),
            'OOF_RMSE': round(rmse, 4),
            'OOF_R2': round(r2, 4),
            'Mean_Fold_MAE': round(float(np.mean(f_maes)), 4),
            'Std_Fold_MAE': round(float(np.std(f_maes)), 4),
            'Tier_Accuracy': round(acc, 4),
            'Tier_Macro_F1': round(macro_f1, 4)
        })

    summary_df = pd.DataFrame(summary_list).sort_values(by='OOF_MAE')
    
    print("\n======================================================================")
    print(" PHASES 3 & 4 FINAL COMPARISON TABLE (ALL BASE MODELS & ENSEMBLES)")
    print("======================================================================")
    print(summary_df.to_string(index=False))

    # Save summary and predictions
    os.makedirs('spatial_cv_results', exist_ok=True)
    summary_df.to_csv('spatial_cv_results/ensemble_phases_3_4_comparison.csv', index=False)
    
    with open('spatial_cv_results/learned_weights_folds.json', 'w') as f:
        json.dump(learned_weights_records, f, indent=2)

    print("\nResults saved to spatial_cv_results/ensemble_phases_3_4_comparison.csv")

if __name__ == '__main__':
    run_ensemble_experiments()

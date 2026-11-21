"""
spatial_cv/tune_optuna.py
Phase 2: Bayesian Hyperparameter Optimization with Optuna (TPE vs Random Search).
Strictly uses inner spatial GroupKFold cross-validation on ward_id.
Never touches outer test folds.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import time
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from spatial_cv.data import load_and_engineer_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_tuning(n_trials=25):
    print("======================================================================")
    print(" PHASE 2: BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA TPE)")
    print("======================================================================")
    
    X, y, groups_ward, groups_city, clean_df, feature_cols = load_and_engineer_features()
    
    # We use Fold 1's training wards (48 wards, 24,864 rows) as the development set
    # and partition it into an inner 4-fold GroupKFold for hyperparameter tuning.
    gkf_outer = GroupKFold(n_splits=5)
    train_idx_dev, test_idx_outer = next(gkf_outer.split(X, y, groups=groups_ward))
    
    X_dev = X.iloc[train_idx_dev].reset_index(drop=True)
    y_dev = y.iloc[train_idx_dev].reset_index(drop=True)
    wards_dev = groups_ward.iloc[train_idx_dev].reset_index(drop=True)

    inner_cv = GroupKFold(n_splits=4)
    inner_splits = list(inner_cv.split(X_dev, y_dev, groups=wards_dev))
    print(f"Development Set: {len(X_dev)} samples across {wards_dev.nunique()} wards")
    print(f"Inner Validation: 4-Fold GroupKFold ({len(inner_splits)} folds, ~12 wards per validation split)")

    def evaluate_inner_lgb(params):
        cv_maes = []
        for tr_idx, val_idx in inner_splits:
            X_tr, y_tr = X_dev.iloc[tr_idx], y_dev.iloc[tr_idx]
            X_va, y_va = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = np.clip(model.predict(X_va), 0, None)
            cv_maes.append(mean_absolute_error(y_va, preds))
        return float(np.mean(cv_maes))

    # 1. LightGBM Optimization with TPE
    print("\n--- 1. Optimizing LightGBM with Optuna TPE ---")
    def objective_lgb(trial):
        obj = trial.suggest_categorical('objective', ['regression_l1', 'tweedie', 'poisson'])
        params = {
            'objective': obj,
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        if obj == 'tweedie':
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.2, 1.8)
        return evaluate_inner_lgb(params)

    t0 = time.time()
    study_lgb_tpe = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb_tpe.optimize(objective_lgb, n_trials=n_trials)
    t_tpe = time.time() - t0

    print(f"Optuna TPE completed {n_trials} trials in {t_tpe:.1f}s")
    print(f"Best Inner Spatial MAE (TPE): {study_lgb_tpe.best_value:.4f}")
    print("Best LightGBM Parameters:", study_lgb_tpe.best_params)

    # 2. Comparison with Random Search on LightGBM
    print("\n--- 2. Comparing with Random Search (Equivalent Budget) ---")
    t0 = time.time()
    study_lgb_rnd = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
    study_lgb_rnd.optimize(objective_lgb, n_trials=n_trials)
    t_rnd = time.time() - t0
    print(f"Random Search completed {n_trials} trials in {t_rnd:.1f}s")
    print(f"Best Inner Spatial MAE (Random Search): {study_lgb_rnd.best_value:.4f}")

    # 3. CatBoost Optimization with TPE
    print("\n--- 3. Optimizing CatBoost with Optuna TPE ---")
    def evaluate_inner_cb(params):
        cv_maes = []
        for tr_idx, val_idx in inner_splits:
            X_tr, y_tr = X_dev.iloc[tr_idx], y_dev.iloc[tr_idx]
            X_va, y_va = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
            model = cb.CatBoostRegressor(**params, verbose=0, random_seed=42, thread_count=-1)
            model.fit(X_tr, y_tr)
            preds = np.clip(model.predict(X_va), 0, None)
            cv_maes.append(mean_absolute_error(y_va, preds))
        return float(np.mean(cv_maes))

    def objective_cb(trial):
        loss_fn = trial.suggest_categorical('loss_function', ['MAE', 'Tweedie:variance_power=1.5', 'Poisson'])
        params = {
            'loss_function': loss_fn,
            'iterations': trial.suggest_int('iterations', 150, 350, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        }
        return evaluate_inner_cb(params)

    study_cb_tpe = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_cb_tpe.optimize(objective_cb, n_trials=25)
    print(f"Best Inner Spatial MAE (CatBoost TPE): {study_cb_tpe.best_value:.4f}")
    print("Best CatBoost Parameters:", study_cb_tpe.best_params)

    # Save best parameters to json
    tuning_results = {
        'lightgbm_tpe': {
            'best_inner_mae': round(study_lgb_tpe.best_value, 4),
            'best_params': study_lgb_tpe.best_params,
            'runtime_sec': round(t_tpe, 2)
        },
        'lightgbm_random': {
            'best_inner_mae': round(study_lgb_rnd.best_value, 4),
            'best_params': study_lgb_rnd.best_params,
            'runtime_sec': round(t_rnd, 2)
        },
        'catboost_tpe': {
            'best_inner_mae': round(study_cb_tpe.best_value, 4),
            'best_params': study_cb_tpe.best_params
        }
    }
    
    os.makedirs('spatial_cv_results', exist_ok=True)
    with open('spatial_cv_results/bayesian_tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)

    print("\nTuning results saved to spatial_cv_results/bayesian_tuning_results.json")

if __name__ == '__main__':
    run_tuning()

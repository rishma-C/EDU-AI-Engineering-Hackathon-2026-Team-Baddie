"""
spatial_cv/models.py
Model factory with Scikit-learn Pipeline architecture and leakage-free transformations.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor,
    RandomForestClassifier
)
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

class DengueEnsemblePredictor:
    """
    Production-grade serialized ensemble combining Bayesian-tuned LightGBM (Tweedie)
    and CatBoost (MAE & Tweedie) models via simplex-constrained optimization.
    Provides sub-second local inference and native risk tier mapping.
    """
    def __init__(self, models_dict, weights_dict, feature_names=None):
        self.models_dict = models_dict
        self.weights_dict = weights_dict
        self.feature_names = feature_names

    def _prepare_input(self, X):
        import pandas as pd
        import numpy as np
        if isinstance(X, np.ndarray):
            if self.feature_names is not None and X.shape[1] == len(self.feature_names):
                return pd.DataFrame(X, columns=self.feature_names)
            return pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                missing = [c for c in self.feature_names if c not in X.columns]
                if not missing:
                    return X[self.feature_names]
            return X
        return X

    def predict(self, X):
        """Generates continuous non-negative dengue case predictions."""
        import numpy as np
        X_df = self._prepare_input(X)
        final_preds = np.zeros(len(X_df))
        for m_name, model in self.models_dict.items():
            w = self.weights_dict.get(m_name, 0.0)
            if w > 0:
                p = np.clip(model.predict(X_df), 0, None)
                final_preds += w * p
        return np.clip(final_preds, 0, None)

    def predict_risk_tier(self, X, cutoffs=(0.5, 5.5, 14.5)):
        """Generates operational epidemiological risk tiers (Low, Medium, High, Critical)."""
        import pandas as pd
        import numpy as np
        preds = self.predict(X)
        c1, c2, c3 = cutoffs
        return pd.cut(
            preds,
            bins=[-np.inf, c1, c2, c3, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

def get_regression_models():
    """
    Returns dictionary of candidate regression models encapsulated in Pipelines.
    """
    models = {
        'Dummy Regressor (Mean)': DummyRegressor(strategy='mean'),
        
        'Linear Regression (Ridge/OLS)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', LinearRegression())
        ]),
        
        'Poisson Regression (GLM)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', PoissonRegressor(alpha=1.0, max_iter=400))
        ]),
        
        'K-Nearest Neighbours': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', KNeighborsRegressor(n_neighbors=15, weights='distance'))
        ]),
        
        'Multilayer Perceptron (MLP)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42, early_stopping=True))
        ]),
        
        'SVR (Linear Kernel)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', SVR(kernel='linear', C=1.0, max_iter=3000))
        ]),
        
        'SVR (RBF Kernel)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', SVR(kernel='rbf', C=5.0, epsilon=0.2, max_iter=3000))
        ]),
        
        'SVR (Polynomial Kernel)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', SVR(kernel='poly', degree=2, C=1.0, max_iter=3000))
        ]),
        
        'SVR (Sigmoid Kernel)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', SVR(kernel='sigmoid', C=1.0, max_iter=3000))
        ]),
        
        'AdaBoost Regressor': AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        
        'Extra Trees Regressor': ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        
        'Hist Gradient Boosting': HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=42),
        
        'XGBoost Regressor': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1),
        
        'CatBoost Regressor (MAE)': cb.CatBoostRegressor(loss_function='MAE', iterations=250, learning_rate=0.06, depth=7, l2_leaf_reg=4.4, random_seed=42, verbose=0, thread_count=-1),
        
        'CatBoost Regressor (Tweedie)': cb.CatBoostRegressor(loss_function='Tweedie:variance_power=1.5', iterations=250, learning_rate=0.06, depth=7, l2_leaf_reg=4.4, random_seed=42, verbose=0, thread_count=-1),
        
        'LightGBM Regressor (L1)': lgb.LGBMRegressor(objective='regression_l1', n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1),

        'LightGBM Regressor (Tweedie)': lgb.LGBMRegressor(objective='tweedie', tweedie_variance_power=1.5, n_estimators=350, learning_rate=0.065, num_leaves=23, max_depth=5, min_child_samples=17, subsample=0.613, colsample_bytree=0.834, reg_alpha=0.0054, reg_lambda=0.0356, random_state=42, verbose=-1, n_jobs=-1)
    }
    return models

def get_tuned_ensemble_base_models():
    """
    Returns optimal high-performing base models for ensemble stacking and simplex weighting.
    """
    return {
        'Tuned_LightGBM_Tweedie': lgb.LGBMRegressor(
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
        )
    }

def get_classification_models():
    """
    Returns dictionary of classifiers for direct risk-tier prediction.
    """
    models = {
        'Logistic Regression (Multinomial)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=400, random_state=42))
        ]),
        
        'Gaussian Naive Bayes': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ]),
        
        'KNN Classifier': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=15, weights='distance'))
        ]),
        
        'SVC (RBF Kernel)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=5.0, max_iter=3000, random_state=42))
        ]),
        
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    }
    return models

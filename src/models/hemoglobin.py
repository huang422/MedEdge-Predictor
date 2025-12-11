"""
Hemoglobin (RTHGB) Prediction Model - Regression using LightGBM/XGBoost.
Target: RTHGB_mapDate (Real-Time Hemoglobin Value)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional
import lightgbm as lgb
from xgboost import XGBRegressor

from .base import BaseModel
from ..config.settings import ModelConfig
from ..utils.metrics import RegressionMetrics


class HemoglobinModel(BaseModel):
    """Real-time hemoglobin prediction model for anemia management."""

    def __init__(self, config: Optional[ModelConfig] = None, exclude_zero_mape: bool = True):
        super().__init__("Hemoglobin", config)
        self.exclude_zero_mape = exclude_zero_mape
        self._training_time: float = 0

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_boost_round: Optional[int] = None
    ) -> None:
        """Train using LightGBM with RMSE metric."""
        start_time = time.time()

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Training parameters
        params = self.config.lgbm_regression_params
        num_rounds = num_boost_round or self.config.num_boost_round

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[val_data]
        )

        self._is_trained = True
        self._training_time = time.time() - start_time

        print(f"Training completed in {self._training_time:.1f} seconds")

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """Train using XGBoost as alternative model."""
        start_time = time.time()

        self.feature_names = X_train.columns.tolist()

        # Create and train XGBoost model
        self.model = XGBRegressor(**self.config.xgb_regression_params)

        self.model.fit(
            X_train,
            y_train,
            early_stopping_rounds=self.config.early_stopping_rounds,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        self._is_trained = True
        self._training_time = time.time() - start_time

        print(f"XGBoost training completed in {self._training_time:.1f} seconds")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict hemoglobin values."""
        if not self._is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(self.model, lgb.Booster):
            return self.model.predict(
                X,
                num_iteration=self.model.best_iteration
            )
        else:
            return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model and return RMSE, MSE, MAPE."""
        y_pred = self.predict(X)

        metrics = RegressionMetrics.calculate_all(
            y.values,
            y_pred,
            exclude_zero=self.exclude_zero_mape
        )

        return metrics

    def get_prediction_analysis(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get detailed prediction analysis with error metrics."""
        y_pred = self.predict(X)

        return RegressionMetrics.get_prediction_analysis(y.values, y_pred)

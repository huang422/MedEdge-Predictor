"""
Dry Weight Prediction Model - Regression using LightGBM/XGBoost.
Target: DryWeight_Y (optimal post-dialysis weight in kg)
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


class DryWeightModel(BaseModel):
    """Optimal dry weight prediction based on weight stability patterns."""

    def __init__(self, config: Optional[ModelConfig] = None, stable_count: int = 9, exclude_zero_mape: bool = True):
        super().__init__("DryWeight", config)
        self.stable_count = stable_count
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
        """Predict dry weight values in kg."""
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

        analysis = RegressionMetrics.get_prediction_analysis(y.values, y_pred)
        analysis["predicted_kg"] = y_pred
        analysis["actual_kg"] = y.values

        return analysis

    def calculate_weight_recommendation(
        self,
        predicted_dry_weight: float,
        current_weight: float,
        safety_margin: float = 0.5
    ) -> Dict[str, float]:
        """Calculate fluid removal recommendations with safety margin."""
        weight_to_remove = current_weight - predicted_dry_weight
        safe_removal = max(0, weight_to_remove - safety_margin)

        return {
            "predicted_dry_weight_kg": round(predicted_dry_weight, 2),
            "current_weight_kg": round(current_weight, 2),
            "total_fluid_excess_kg": round(weight_to_remove, 2),
            "recommended_uf_kg": round(safe_removal, 2),
            "safety_margin_kg": safety_margin
        }

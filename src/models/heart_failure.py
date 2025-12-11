"""
Heart Failure Prediction Model - Binary classification using LightGBM/XGBoost.
Target: HF_1 (1 = High Risk, 0 = Low Risk)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

from .base import BaseModel
from ..config.settings import ModelConfig
from ..utils.metrics import ClassificationMetrics


class HeartFailureModel(BaseModel):
    """Heart Failure prediction model with SMOTE for class balancing."""

    def __init__(self, config: Optional[ModelConfig] = None, use_smote: bool = True):
        super().__init__("HeartFailure", config)
        self.use_smote = use_smote
        self.threshold = self.config.classification_threshold
        self._training_time: float = 0

    def _apply_smote(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Apply SMOTE to balance classes."""
        print(f"Class distribution before SMOTE: {Counter(y)}")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"Class distribution after SMOTE: {Counter(y_resampled)}")
        return X_resampled, y_resampled

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_boost_round: Optional[int] = None
    ) -> None:
        """Train using LightGBM with optional SMOTE for class balancing."""
        start_time = time.time()

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Apply SMOTE if enabled
        if self.use_smote:
            X_train_balanced, y_train_balanced = self._apply_smote(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_balanced, label=y_train_balanced)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Training parameters
        params = self.config.lgbm_classification_params
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

        # Apply SMOTE if enabled
        if self.use_smote:
            X_train_balanced, y_train_balanced = self._apply_smote(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Create and train XGBoost model
        self.model = XGBClassifier(**self.config.xgb_classification_params)

        self.model.fit(
            X_train_balanced,
            y_train_balanced,
            early_stopping_rounds=self.config.early_stopping_rounds,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        self._is_trained = True
        self._training_time = time.time() - start_time

        print(f"XGBoost training completed in {self._training_time:.1f} seconds")

    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """Make predictions. Returns probabilities if return_proba=True."""
        if not self._is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(self.model, lgb.Booster):
            proba = self.model.predict(
                X,
                num_iteration=self.model.best_iteration
            )
        else:
            proba = self.model.predict_proba(X)[:, 1]

        if return_proba:
            return proba

        return (proba > self.threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model and return classification metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict(X, return_proba=True)

        metrics = ClassificationMetrics.calculate_all(y, y_pred, y_proba)

        return metrics

    def get_misclassified(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get samples that were misclassified."""
        y_pred = self.predict(X)

        output = pd.DataFrame({
            "actual": y.values,
            "predicted": y_pred
        })

        misclassified = output[output["actual"] != output["predicted"]]
        print(f"Misclassified samples: {len(misclassified)}")

        return misclassified

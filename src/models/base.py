"""
Base Model Module - Abstract base class for all prediction models.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import lightgbm as lgb

from ..config.settings import ModelConfig


class BaseModel(ABC):
    """Abstract base class for medical prediction models."""

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None):
        self.model_name = model_name
        self.config = config or ModelConfig()
        self.model = None
        self.feature_names: list = []
        self._is_trained = False

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        pass

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        if isinstance(self.model, lgb.Booster):
            self.model.save_model(filepath)
        else:
            import joblib
            joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        if filepath.endswith(".txt"):
            self.model = lgb.Booster(model_file=filepath)
        else:
            import joblib
            self.model = joblib.load(filepath)
        self._is_trained = True

    def get_feature_importance(self, importance_type: str = "gain") -> Tuple[np.ndarray, np.ndarray]:
        """Get feature importance from trained model."""
        if not self._is_trained or self.model is None:
            raise ValueError("Model not trained.")

        if isinstance(self.model, lgb.Booster):
            names = self.model.feature_name()
            importance = self.model.feature_importance(importance_type=importance_type)
        else:
            names = self.feature_names
            importance = self.model.feature_importances_

        return np.array(names), importance

    def get_top_features(self, top_n: int = 20, importance_type: str = "gain") -> pd.DataFrame:
        """Get top N most important features."""
        names, importance = self.get_feature_importance(importance_type)
        total = importance.sum()
        percentages = 100 * importance / total

        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
            "percentage": percentages
        }).sort_values("importance", ascending=False)

        return df.head(top_n)

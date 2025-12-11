"""
Evaluation Metrics - Classification and regression metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_percentage_error
)


class ClassificationMetrics:
    """Metrics for classification models (Heart Failure, IDH)."""

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate accuracy, precision, recall, F1, and AUC."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }

        if y_prob is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["auc"] = np.nan

        return metrics

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Return confusion matrix with TN, FP, FN, TP labels."""
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            labeled = {
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1]
            }
        else:
            labeled = {}

        return cm, labeled

    @staticmethod
    def print_report(metrics: Dict[str, float], model_name: str = "Model") -> None:
        """Print formatted metrics report."""
        print(f"\n{'='*50}")
        print(f"{model_name} - Classification Metrics")
        print(f"{'='*50}")
        for name, value in metrics.items():
            if not np.isnan(value):
                print(f"{name.capitalize():>12}: {value*100:.2f}%")
        print(f"{'='*50}\n")


class RegressionMetrics:
    """Metrics for regression models (Hemoglobin, Dry Weight)."""

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        exclude_zero: bool = True
    ) -> Dict[str, float]:
        """Calculate RMSE, MSE, and MAPE."""
        metrics = {
            "rmse": RegressionMetrics.root_mean_squared_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred)
        }

        # MAPE calculation (excluding zeros if specified)
        if exclude_zero:
            mask = y_true != 0
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred

        if len(y_true_filtered) > 0:
            metrics["mape"] = mean_absolute_percentage_error(
                y_true_filtered, y_pred_filtered
            )
        else:
            metrics["mape"] = np.nan

        return metrics

    @staticmethod
    def get_prediction_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """Return DataFrame with actual, predicted, and difference columns."""
        return pd.DataFrame({
            "actual": y_true,
            "predicted": y_pred,
            "difference": y_true - y_pred,
            "abs_difference": np.abs(y_true - y_pred)
        })

    @staticmethod
    def print_report(metrics: Dict[str, float], model_name: str = "Model") -> None:
        """Print formatted metrics report."""
        print(f"\n{'='*50}")
        print(f"{model_name} - Regression Metrics")
        print(f"{'='*50}")
        for name, value in metrics.items():
            if not np.isnan(value):
                if name == "mape":
                    print(f"{name.upper():>8}: {value*100:.2f}%")
                else:
                    print(f"{name.upper():>8}: {value:.4f}")
        print(f"{'='*50}\n")

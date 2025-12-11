"""
Visualization Module - Plots for model analysis and interpretation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelVisualizer:
    """Visualization utilities for model analysis."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix heatmap."""
        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        counts = [f"{v:0.0f}" for v in confusion_matrix.flatten()]
        annotations = np.array([
            f"{label}\n{count}"
            for label, count in zip(labels.flatten(), counts)
        ]).reshape(2, 2)

        plt.figure(figsize=(6, 6))
        sns.heatmap(
            confusion_matrix,
            square=True,
            annot=annotations,
            fmt="",
            linecolor="white",
            cmap="RdBu",
            linewidths=1.5,
            cbar=False
        )
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.title(title, fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 20,
        importance_type: str = "gain",
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Plot horizontal bar chart of top N features."""
        # Calculate percentage importance
        total = importance_values.sum()
        percentages = 100 * importance_values / total

        # Create DataFrame
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values,
            "percentage": percentages
        }).sort_values("importance", ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=self.figsize)
        color = "lightgreen" if importance_type == "gain" else "skyblue"

        plt.barh(df["feature"], df["percentage"], color=color)
        plt.xlabel(f"Importance ({importance_type}) %", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title(title or f"Top {top_n} Features by {importance_type.capitalize()} Importance",
                 fontsize=14)
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return df

    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Prediction Distribution",
        save_path: Optional[str] = None
    ) -> None:
        """Plot histogram comparing actual vs predicted values."""
        plt.figure(figsize=self.figsize)

        sns.histplot(
            np.round(y_true, 1),
            color="blue",
            label="Actual",
            kde=False,
            stat="count",
            alpha=0.6
        )
        sns.histplot(
            np.round(y_pred, 1),
            color="red",
            label="Predicted",
            kde=False,
            stat="count",
            alpha=0.6
        )

        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_prediction_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Prediction Error",
        save_path: Optional[str] = None
    ) -> None:
        """Plot prediction error over samples."""
        difference = y_true - y_pred

        plt.figure(figsize=self.figsize)
        plt.plot(range(len(difference)), difference, color="green", linewidth=1)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)

        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("Error (Actual - Predicted)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_kde_by_prediction(
        self,
        feature_values: np.ndarray,
        predictions: np.ndarray,
        feature_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """Plot KDE distribution of feature split by prediction class."""
        df = pd.DataFrame({
            "feature": feature_values,
            "prediction": predictions
        })

        # Filter valid data
        valid_df = df[df["feature"] > 0].dropna()

        plt.figure(figsize=self.figsize)

        sns.kdeplot(
            data=valid_df[valid_df["prediction"] == 0],
            x="feature",
            label="Negative",
            fill=True,
            common_norm=False,
            alpha=0.5
        )
        sns.kdeplot(
            data=valid_df[valid_df["prediction"] == 1],
            x="feature",
            label="Positive",
            fill=True,
            common_norm=False,
            alpha=0.5
        )

        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title(f"{feature_name} Distribution by Prediction", fontsize=14)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_shap_summary(
        shap_values: np.ndarray,
        features: pd.DataFrame,
        top_n: int = 10,
        title: str = "SHAP Feature Importance"
    ) -> None:
        """Plot SHAP summary for model interpretation."""
        try:
            import shap

            # Get top features by mean absolute SHAP value
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
            feature_names = features.columns[top_indices].tolist()

            plt.title(title)
            shap.summary_plot(
                shap_values[:, top_indices],
                features.iloc[:, top_indices],
                feature_names=feature_names,
                show=True
            )
        except ImportError:
            print("SHAP library not installed. Run: pip install shap")

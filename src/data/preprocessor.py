"""
Data Preprocessing Module for medical prediction models.
Handles missing values, feature scaling, skewness transformation, and encoding.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from scipy.stats import skew

from ..config.settings import DataConfig


class DataPreprocessor:
    """Data preprocessor for medical prediction models."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method="yeo-johnson")
        self._is_fitted = False
        self._skewed_columns: List[str] = []
        self._numeric_columns: List[str] = []

    def handle_missing_values(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Drop columns with missing ratio above threshold."""
        threshold = threshold or self.config.missing_threshold
        df = df.replace("NULL", np.nan)
        missing_ratios = df.isnull().mean()
        columns_to_keep = missing_ratios[missing_ratios <= threshold].index
        return df[columns_to_keep].copy()

    def transform_skewed_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply Yeo-Johnson transformation to highly skewed features."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        skew_values = df[columns].apply(lambda x: skew(x.dropna()))
        threshold = self.config.skewness_threshold
        high_skew = skew_values[(skew_values > threshold) | (skew_values < -threshold)]
        skewed_cols = high_skew.index.tolist()

        if skewed_cols:
            df[skewed_cols] = self.power_transformer.fit_transform(df[skewed_cols])
            self._skewed_columns = skewed_cols

        return df

    def scale_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply StandardScaler to numeric features."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self._numeric_columns = columns
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def encode_categorical(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply one-hot encoding to categorical columns."""
        columns = columns or self.config.categorical_columns
        columns = [col for col in columns if col in df.columns]

        if columns:
            df = pd.get_dummies(df, columns=columns, dtype=int)

        return df

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove special characters from column names for model compatibility."""
        special_chars = "[]{}:,\"'/-()$"
        replace_char = "9"

        new_names = {}
        for col in df.columns:
            if any(char in col for char in special_chars):
                new_name = col
                for char in special_chars:
                    new_name = new_name.replace(char, replace_char)
                new_names[col] = new_name

        if new_names:
            df = df.rename(columns=new_names)

        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full preprocessing pipeline.
        Returns tuple of (features DataFrame, target Series).
        """
        exclude_columns = exclude_columns or []

        df = self.handle_missing_values(df)

        feature_cols = [col for col in df.columns
                       if col != target_column and col not in exclude_columns]

        X = df[feature_cols].copy()
        y = df[target_column].copy()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        X = self.transform_skewed_features(X, numeric_cols)
        X = self.scale_features(X, numeric_cols)
        X = self.encode_categorical(X)
        X = self.clean_column_names(X)

        self._is_fitted = True
        return X, y


class KNNDataImputer:
    """KNN-based imputer for dashboard data preprocessing."""

    def __init__(self, n_neighbors: int = 5):
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def impute(
        self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Impute missing values using KNN."""
        exclude_columns = exclude_columns or []

        excluded_df = df[exclude_columns].copy()
        numeric_df = df.drop(columns=exclude_columns)

        imputed_values = self.imputer.fit_transform(numeric_df)
        imputed_df = pd.DataFrame(imputed_values, columns=numeric_df.columns)

        # Reset index for merge
        excluded_df = excluded_df.reset_index(drop=True)
        imputed_df = imputed_df.reset_index(drop=True)

        return pd.concat([excluded_df, imputed_df], axis=1)

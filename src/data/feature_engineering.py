"""
Feature Engineering Module for medical prediction models.
Handles dry weight stability labeling and time-based train/test splitting.
"""

import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split

from ..config.settings import DataConfig


class FeatureEngineer:
    """Feature engineering for medical prediction models."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def calculate_dry_weight_label(
        self, df: pd.DataFrame, stable_count: int = 9
    ) -> pd.DataFrame:
        """
        Calculate dry weight target labels based on stability criteria.
        A patient's dry weight is stable if it remains constant across
        consecutive dialysis sessions.
        """
        df = df.sort_values(by=["MEDICALID", "PTIMESTAMP", "NumberState"])
        df["DryWeight_Y"] = pd.NA
        df["PostWeight_pre"] = pd.NA

        # Get last non-null values per session
        df["DryWeight_last"] = df.groupby(["MEDICALID", "NumberState"])[
            "DryWeight"
        ].transform(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else pd.NA)

        df["PostWeight_pre_last"] = df.groupby(["MEDICALID", "NumberState"])[
            "PostWeight"
        ].transform(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else pd.NA)

        df["DryWeight_last"] = pd.to_numeric(df["DryWeight_last"], errors="coerce")
        df["PostWeight_pre_last"] = pd.to_numeric(df["PostWeight_pre_last"], errors="coerce")

        def check_stability(group: pd.DataFrame) -> pd.DataFrame:
            """Check weight stability for a patient group."""
            last_rows = group.drop_duplicates(subset="NumberState", keep="last")

            # Rolling window to check if all values are identical
            last_rows["is_stable"] = (
                last_rows["DryWeight_last"]
                .rolling(window=stable_count, min_periods=stable_count)
                .apply(lambda x: len(set(x)) == 1, raw=True)
            )

            stable_rows = last_rows[last_rows["is_stable"] == 1]

            for idx in stable_rows.index:
                stable_value = last_rows.loc[idx, "DryWeight_last"]
                max_state = last_rows.loc[idx, "NumberState"]
                group.loc[
                    (group["NumberState"] <= max_state) & (group["DryWeight_Y"].isna()),
                    "DryWeight_Y"
                ] = stable_value

            group["PreWeight"] = group["PreWeight"].ffill()

            last_rows["PostWeight_pre"] = last_rows["PostWeight_pre_last"].shift(1).ffill()
            last_rows["PostWeight_pre"] = last_rows["PostWeight_pre"].fillna(
                last_rows["PostWeight_pre_last"]
            )

            group = group.merge(
                last_rows[["NumberState", "PostWeight_pre"]],
                on="NumberState",
                suffixes=("", "_update")
            )
            group["PostWeight_pre"] = group["PostWeight_pre_update"]
            group.drop(columns=["PostWeight_pre_update"], inplace=True)

            return group

        df = df.groupby("MEDICALID", group_keys=False).apply(check_stability)
        df.dropna(subset=["DryWeight_Y"], inplace=True)
        df.drop(columns=["DryWeight_last", "PostWeight_pre_last", "DryWeight"], inplace=True)
        df["DryWeight_Y"] = pd.to_numeric(df["DryWeight_Y"], errors="coerce")

        return df

    def split_by_time(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "PTIMESTAMP",
        train_month: int = 12,
        test_month: int = 11
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by month for time-based validation."""
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        train_data = df[df[timestamp_column].dt.month == train_month]
        test_data = df[df[timestamp_column].dt.month == test_month]
        return train_data, test_data

    def prepare_time_split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        timestamp_column: str = "PTIMESTAMP",
        train_month: int = 12,
        test_month: int = 11,
        exclude_columns: Optional[list] = None
    ) -> Dict:
        """
        Prepare data with time-based train/test split.
        Returns dict with X_train, X_val, X_test, y_train, y_val, y_test.
        """
        exclude_columns = exclude_columns or [timestamp_column]

        train_data, test_data = self.split_by_time(
            df, timestamp_column, train_month, test_month
        )

        feature_cols = [col for col in df.columns
                       if col != target_column and col not in exclude_columns]

        X_train_full = train_data[feature_cols]
        y_train_full = train_data[target_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        X_test = test_data[feature_cols]
        y_test = test_data[target_column]

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": feature_cols
        }

"""
Dashboard Data Merger Module

This module handles merging prediction results from all models
into a single dataset for dashboard visualization.
"""

import pandas as pd
from typing import List, Optional
from sklearn.impute import KNNImputer


class PredictionDataMerger:
    """
    Merges prediction outputs from multiple models for dashboard display.

    Combines predictions from:
    - Heart Failure model (classification)
    - Hemoglobin model (regression)
    - Dry Weight model (regression)
    - IDH model (classification)
    """

    def __init__(self, n_neighbors: int = 5):
        """
        Initialize the merger.

        Args:
            n_neighbors: Number of neighbors for KNN imputation
        """
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def merge_predictions(
        self,
        df_hf: pd.DataFrame,
        df_rthgb: pd.DataFrame,
        df_dw: pd.DataFrame,
        df_idh: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge prediction DataFrames from all models.

        Args:
            df_hf: Heart Failure predictions
            df_rthgb: Hemoglobin predictions
            df_dw: Dry Weight predictions
            df_idh: IDH predictions (optional)

        Returns:
            Merged DataFrame with all predictions
        """
        # Merge on common keys
        merged = df_hf.merge(df_rthgb, on=["MEDICALID", "PTIMESTAMP"], how="left")
        merged = merged.merge(df_dw, on=["MEDICALID", "PTIMESTAMP"], how="left")

        if df_idh is not None:
            idh_selected = df_idh[["MEDICALID", "PTIMESTAMP", "Pred_IDH", "Pred_IDH_prob"]]
            merged = merged.merge(idh_selected, on=["MEDICALID", "PTIMESTAMP"], how="left")

        # Sort and clean
        merged = merged.sort_values(by=["MEDICALID", "PTIMESTAMP"])
        merged = merged.dropna(subset=["Pred_DW"])

        # Format datetime columns
        if "RTHGB_DATE" in merged.columns:
            merged["RTHGB_DATE"] = pd.to_datetime(merged["RTHGB_DATE"]).dt.strftime("%Y-%m-%d %H:%M")
        merged["PTIMESTAMP"] = pd.to_datetime(merged["PTIMESTAMP"]).dt.strftime("%Y-%m-%d %H:%M")

        return merged

    def select_dashboard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select columns needed for dashboard display.

        Args:
            df: Merged DataFrame

        Returns:
            DataFrame with selected columns
        """
        columns_to_keep = [
            "MEDICALID", "PTIMESTAMP",
            "Pred_HF", "Pred_HF_prob",
            "Pred_RTHGB", "Pred_DW",
            "TAST", "FERRITIN",
            "ferr_outlier_H", "ferr_date_H",
            "LatestESA", "LatestESA_DATE", "TotalOfESA",
            "A_TMP", "A_VENOUSPRESSURE", "A_ARTERIALPRESSURE",
            "A_TOTALUF", "A_D_TEMPERATURE", "A_BICARBONATEADJUSTMENT",
            "RTHGB_DATE", "Pred_IDH", "Pred_IDH_prob"
        ]

        # Filter to existing columns
        existing_cols = [col for col in columns_to_keep if col in df.columns]
        return df[existing_cols].copy()

    def anonymize_patient_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create anonymous patient IDs for display.

        Args:
            df: DataFrame with MEDICALID column

        Returns:
            DataFrame with IDs column (A01, A02, etc.)
        """
        unique_ids = df["MEDICALID"].unique()
        id_mapping = {med_id: f"A{i+1:02d}" for i, med_id in enumerate(unique_ids)}
        df["IDs"] = df["MEDICALID"].map(id_mapping)
        return df

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Impute missing values using KNN.

        Args:
            df: Input DataFrame
            exclude_columns: Columns to exclude from imputation

        Returns:
            DataFrame with imputed values
        """
        exclude_columns = exclude_columns or ["MEDICALID", "PTIMESTAMP", "RTHGB_DATE", "IDs"]

        # Separate excluded and numeric columns
        excluded_df = df[exclude_columns].copy()
        numeric_df = df.drop(columns=exclude_columns)

        # Impute
        imputed_values = self.imputer.fit_transform(numeric_df)
        imputed_df = pd.DataFrame(imputed_values, columns=numeric_df.columns)

        # Merge back
        imputed_df["index"] = range(len(imputed_df))
        excluded_df["index"] = range(len(excluded_df))

        result = pd.merge(excluded_df, imputed_df, on="index")
        result = result.drop(columns=["index"])

        return result

    def prepare_dashboard_data(
        self,
        df_hf: pd.DataFrame,
        df_rthgb: pd.DataFrame,
        df_dw: pd.DataFrame,
        df_idh: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Full pipeline to prepare data for dashboard.

        Args:
            df_hf: Heart Failure predictions
            df_rthgb: Hemoglobin predictions
            df_dw: Dry Weight predictions
            df_idh: IDH predictions (optional)
            output_path: Path to save output CSV (optional)

        Returns:
            Prepared DataFrame ready for dashboard
        """
        # Merge
        merged = self.merge_predictions(df_hf, df_rthgb, df_dw, df_idh)

        # Select columns
        selected = self.select_dashboard_columns(merged)

        # Anonymize
        anonymized = self.anonymize_patient_ids(selected)

        # Impute
        imputed = self.impute_missing_values(anonymized)

        # Add index column for dashboard
        imputed["index"] = range(len(imputed))

        # Save if path provided
        if output_path:
            imputed.to_csv(output_path, index=False)
            print(f"Dashboard data saved to {output_path}")

        return imputed


# Convenience function
def prepare_dashboard_data(
    hf_path: str,
    rthgb_path: str,
    dw_path: str,
    idh_path: Optional[str] = None,
    output_path: str = "./data/Pred_all.csv"
) -> pd.DataFrame:
    """
    Prepare dashboard data from CSV files.

    Args:
        hf_path: Path to Heart Failure predictions CSV
        rthgb_path: Path to Hemoglobin predictions CSV
        dw_path: Path to Dry Weight predictions CSV
        idh_path: Path to IDH predictions CSV (optional)
        output_path: Path to save merged output

    Returns:
        Prepared DataFrame
    """
    df_hf = pd.read_csv(hf_path)
    df_rthgb = pd.read_csv(rthgb_path)
    df_dw = pd.read_csv(dw_path)
    df_idh = pd.read_csv(idh_path) if idh_path else None

    merger = PredictionDataMerger()
    return merger.prepare_dashboard_data(
        df_hf, df_rthgb, df_dw, df_idh, output_path
    )

"""
Configuration Settings for Medical Prediction System

This module contains all configuration parameters for models, data processing,
and dashboard settings. Database credentials and sensitive information are
excluded for security purposes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """Configuration for data processing parameters."""

    # Missing value threshold (columns with >50% missing are dropped)
    missing_threshold: float = 0.5

    # Skewness threshold for power transformation
    skewness_threshold: float = 1.0

    # Train/validation split ratio
    test_size: float = 0.2
    random_state: int = 42

    # Columns to exclude from feature set (identifiers, targets, etc.)
    id_columns: List[str] = field(default_factory=lambda: ["MEDICALID", "PTIMESTAMP"])

    # Columns requiring one-hot encoding
    categorical_columns: List[str] = field(default_factory=lambda: [
        "A_ISARTERIALPRESSUREALARM",
        "A_ISTMPALARM",
        "A_ISBLOODPRESSUREALARM",
        "A_ISPULSEALARM",
        "A_ALARMSTATUSFLAGS"
    ])

    # Columns to drop (date/outlier columns)
    columns_to_drop: List[str] = field(default_factory=lambda: [
        "cci", "StartTime",
        "hba1c_date_H", "hba1c_outlier_H", "hba1c_date_L", "hba1c_outlier_L",
        "na_date_H", "na_outlier_H", "na_date_L", "na_outlier_L",
        "k_date_H", "k_outlier_H", "k_date_L", "k_outlier_L",
        "ca_date_H", "ca_outlier_H", "ca_date_L", "ca_outlier_L",
        "bun_date_H", "bun_outlier_H", "bun_date_L", "bun_outlier_L",
        "crea_date_H", "crea_outlier_H", "crea_date_L", "crea_outlier_L",
        "alb_date_H", "alb_outlier_H", "alb_date_L", "alb_outlier_L",
        "egfr_date_H", "egfr_outlier_H", "egfr_date_L", "egfr_outlier_L",
        "ipth_date_H", "ipth_outlier_H", "ipth_date_L", "ipth_outlier_L",
        "ferr_date_H", "ferr_outlier_H", "ferr_date_L", "ferr_outlier_L",
        "alum_date_H", "alum_outlier_H", "alum_date_L", "alum_outlier_L",
        "nbnp_date_H", "nbnp_outlier_H", "nbnp_date_L", "nbnp_outlier_L"
    ])


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""

    # LightGBM parameters for classification (Heart Failure)
    lgbm_classification_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "verbose_eval": 100
    })

    # LightGBM parameters for regression (Hemoglobin/Dry Weight)
    lgbm_regression_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "verbose_eval": 100
    })

    # XGBoost parameters for classification
    xgb_classification_params: Dict[str, Any] = field(default_factory=lambda: {
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_estimators": 2000
    })

    # XGBoost parameters for regression
    xgb_regression_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 500,
        "eval_metric": "rmse"
    })

    # Training parameters
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100

    # Classification threshold
    classification_threshold: float = 0.5

    # Model save paths
    model_save_dir: str = "weights"


@dataclass
class DashboardConfig:
    """Configuration for real-time dashboard."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False

    # Auto-refresh interval (milliseconds)
    refresh_interval: int = 3000

    # Display settings
    time_series_window: int = 10  # Number of data points to show

    # Risk thresholds
    hf_risk_threshold: float = 0.5  # Heart failure risk threshold
    idh_risk_threshold: float = 30.0  # Intradialytic hypotension risk threshold

    # Chart colors
    colors: Dict[str, str] = field(default_factory=lambda: {
        "background": "black",
        "text": "white",
        "high_risk": "red",
        "low_risk": "#77DD77",
        "tsat_gauge": "darkorange",
        "ferritin_gauge": "green",
        "triangle": "brown"
    })

    # Chart titles (English)
    chart_titles: Dict[str, str] = field(default_factory=lambda: {
        "A_TMP": "TMP (Transmembrane Pressure)",
        "A_VENOUSPRESSURE": "Venous Pressure",
        "A_ARTERIALPRESSURE": "Arterial Pressure",
        "A_TOTALUF": "UF Volume (Ultrafiltration)",
        "A_D_TEMPERATURE": "Actual Blood Temperature",
        "A_BICARBONATEADJUSTMENT": "Bicarbonate Adjustment"
    })


# Default configurations
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DASHBOARD_CONFIG = DashboardConfig()

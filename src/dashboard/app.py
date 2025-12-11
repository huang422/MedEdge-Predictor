"""
Real-time Medical Prediction Dashboard

This module implements an interactive dashboard for visualizing real-time
predictions from the medical prediction models. Built with Plotly Dash
for deployment on NVIDIA Jetson Orin edge devices.

Features:
- Real-time auto-refresh (3-second intervals)
- Patient ID selection with auto-carousel mode
- Gauge charts for TSAT and FERRITIN indicators
- Time-series charts for dialysis session parameters
- Risk level visualization with color coding
- Clinical decision support recommendations
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from typing import Optional
import sys
import os

# Handle both module and direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.config.settings import DashboardConfig
else:
    from ..config.settings import DashboardConfig


def create_app(
    data_path: str = "./data/Pred_all.csv",
    config: Optional[DashboardConfig] = None
) -> dash.Dash:
    """
    Create and configure the Dash application.

    Args:
        data_path: Path to the prediction data CSV file
        config: DashboardConfig object with display settings

    Returns:
        Configured Dash application instance
    """
    config = config or DashboardConfig()

    # Load prediction data
    df = pd.read_csv(data_path)
    df["PTIMESTAMP"] = pd.to_datetime(df["PTIMESTAMP"])
    df["RTHGB_DATE"] = pd.to_datetime(df["RTHGB_DATE"])
    df = df.sort_values(by=["IDs", "PTIMESTAMP"], ascending=[True, True])

    # Get first record per patient for dropdown
    df_first = df.groupby("IDs").first().reset_index()

    # Initialize Dash app
    app = dash.Dash(__name__)

    # Common styles
    common_style = {
        "textAlign": "center",
        "fontSize": 22,
        "color": config.colors["text"],
        "padding": 0,
        "margin": 0,
        "flex": "1",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "fontWeight": "bold",
        "white-space": "pre-line",
        "height": "100%"
    }

    # Layout definition
    app.layout = html.Div([
        # Header
        html.H2(
            "AI Medical Prediction Dashboard",
            style={
                "textAlign": "center",
                "color": config.colors["text"],
                "height": "3vh",
                "padding-top": "2vh",
                "fontWeight": "bold"
            }
        ),

        # Top row - Controls and key metrics
        html.Div([
            # Patient ID dropdown
            dcc.Dropdown(
                id="id-dropdown",
                options=[{"label": "Auto", "value": "auto"}] +
                        [{"label": f"{row['IDs']}", "value": row["IDs"]}
                         for _, row in df_first.iterrows()],
                value="auto",
                style={
                    "fontSize": 16,
                    "height": "100%",
                    "width": "80px",
                    "fontWeight": "bold"
                },
                placeholder="Select ID",
                searchable=False,
                clearable=False
            ),

            # Patient ID display
            html.Div(
                id="medical-id-display",
                style={
                    "textAlign": "center",
                    "fontSize": 22,
                    "fontWeight": "bold",
                    "color": config.colors["text"],
                    "height": "100%"
                }
            ),

            # Heart Failure prediction
            html.Div([
                html.Div(
                    "HF (Heart Failure Risk): ",
                    style={
                        "textAlign": "center",
                        "fontSize": 22,
                        "fontWeight": "bold",
                        "color": config.colors["text"],
                        "height": "100%"
                    }
                ),
                html.Div(
                    id="hf-display",
                    style={
                        "textAlign": "center",
                        "fontSize": 22,
                        "fontWeight": "bold",
                        "height": "100%"
                    }
                )
            ], style={"display": "flex", "flexDirection": "row", "justifyContent": "center"}),

            # Hemoglobin prediction
            html.Div(
                id="hgb-display",
                style={
                    "textAlign": "center",
                    "fontSize": 22,
                    "fontWeight": "bold",
                    "color": config.colors["text"],
                    "height": "100%"
                }
            ),

            # Dry Weight prediction
            html.Div(
                id="delta-weight-display",
                style={
                    "textAlign": "center",
                    "fontSize": 22,
                    "fontWeight": "bold",
                    "color": config.colors["text"],
                    "height": "100%"
                }
            ),

            # IDH prediction
            html.Div([
                html.Div(
                    "IDH (Hypotension Risk): ",
                    style={
                        "textAlign": "center",
                        "fontSize": 22,
                        "fontWeight": "bold",
                        "color": config.colors["text"],
                        "height": "100%"
                    }
                ),
                html.Div(
                    id="idh-display",
                    style={
                        "textAlign": "center",
                        "fontSize": 22,
                        "fontWeight": "bold",
                        "height": "100%"
                    }
                )
            ], style={"display": "flex", "flexDirection": "row", "justifyContent": "center"})

        ], style={
            "display": "flex",
            "justifyContent": "space-around",
            "backgroundColor": config.colors["background"],
            "height": "2vh",
            "padding-bottom": "1.5vh"
        }),

        # Middle row - Gauges, metrics, and triangle chart
        html.Div([
            # Gauge charts column
            html.Div([
                dcc.Graph(id="gauge-graph-TSAT", style={"height": "50%", "width": "100%"}),
                dcc.Graph(id="gauge-graph-FERRITIN", style={"height": "50%", "width": "100%"})
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "flex": "1",
                "backgroundColor": config.colors["background"],
                "padding": 0,
                "margin": 0,
                "height": "100%",
                "width": "40%"
            }),

            # Clinical metrics column
            html.Div([
                html.Div(id="ferr_outlier_H", style=common_style),
                html.Div(id="ferr_date_H", style=common_style),
                html.Div(id="LatestESA", style=common_style),
                html.Div(id="LatestESA_DATE", style=common_style),
                html.Div(
                    id="doctor_txt",
                    style={
                        "textAlign": "center",
                        "fontSize": 16,
                        "color": config.colors["text"],
                        "flex": "1",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "height": "100%"
                    }
                )
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "flex": "1",
                "backgroundColor": config.colors["background"],
                "height": "100%",
                "width": "40%"
            }),

            # Triangle chart column
            html.Div([
                dcc.Graph(id="triangle-graph", style={"height": "100%", "width": "100%"})
            ], style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "space-around",
                "flex": "1",
                "backgroundColor": config.colors["background"],
                "padding": 0,
                "margin": 0,
                "height": "100%",
                "width": "40%"
            })
        ], style={
            "display": "flex",
            "flexDirection": "row",
            "justifyContent": "space-around",
            "backgroundColor": config.colors["background"],
            "width": "100%",
            "height": "60vh"
        }),

        # Time series charts - Row 1
        html.Div([
            dcc.Graph(id="A_TMP", style={"flex": "1", "width": "33.33%", "height": "100%"}),
            dcc.Graph(id="A_VENOUSPRESSURE", style={"flex": "1", "width": "33.33%", "height": "100%"}),
            dcc.Graph(id="A_ARTERIALPRESSURE", style={"flex": "1", "width": "33.33%", "height": "100%"})
        ], style={
            "display": "flex",
            "flexDirection": "row",
            "justifyContent": "space-around",
            "backgroundColor": config.colors["background"],
            "height": "15vh",
            "padding": 0,
            "margin-bottom": "1vh"
        }),

        # Time series charts - Row 2
        html.Div([
            dcc.Graph(id="A_TOTALUF", style={"flex": "1", "width": "33.33%", "height": "100%"}),
            dcc.Graph(id="A_D_TEMPERATURE", style={"flex": "1", "width": "33.33%", "height": "100%"}),
            dcc.Graph(id="A_BICARBONATEADJUSTMENT", style={"flex": "1", "width": "33.33%", "height": "100%"})
        ], style={
            "display": "flex",
            "flexDirection": "row",
            "justifyContent": "space-around",
            "backgroundColor": config.colors["background"],
            "height": "15vh",
            "padding": 0,
            "margin": 0,
            "margin-bottom": "1vh"
        }),

        # Auto-refresh interval
        dcc.Interval(
            id="interval-component",
            interval=config.refresh_interval,
            n_intervals=0
        )
    ], style={
        "backgroundColor": config.colors["background"],
        "height": "100vh",
        "width": "100%",
        "margin": 0,
        "padding": 0
    })

    # Register callbacks
    _register_callbacks(app, df, config)

    return app


def _register_callbacks(app: dash.Dash, df: pd.DataFrame, config: DashboardConfig):
    """
    Register all dashboard callbacks.

    Args:
        app: Dash application instance
        df: Prediction data DataFrame
        config: Dashboard configuration
    """

    @app.callback(
        [
            Output("medical-id-display", "children"),
            Output("hf-display", "children"),
            Output("hgb-display", "children"),
            Output("delta-weight-display", "children"),
            Output("idh-display", "children"),
            Output("gauge-graph-TSAT", "figure"),
            Output("gauge-graph-FERRITIN", "figure"),
            Output("ferr_outlier_H", "children"),
            Output("ferr_date_H", "children"),
            Output("LatestESA", "children"),
            Output("LatestESA_DATE", "children"),
            Output("triangle-graph", "figure"),
            Output("A_TMP", "figure"),
            Output("A_VENOUSPRESSURE", "figure"),
            Output("A_ARTERIALPRESSURE", "figure"),
            Output("A_TOTALUF", "figure"),
            Output("A_D_TEMPERATURE", "figure"),
            Output("A_BICARBONATEADJUSTMENT", "figure"),
            Output("doctor_txt", "children")
        ],
        [
            Input("id-dropdown", "value"),
            Input("interval-component", "n_intervals")
        ]
    )
    def update_dashboard(selected_id, n_intervals):
        """Main callback to update all dashboard components."""

        # Select patient data
        if selected_id == "auto":
            selected_row = df.iloc[n_intervals % len(df)]
        else:
            df_selected = df[df["IDs"] == selected_id]
            row_index = n_intervals % len(df_selected)
            selected_row = df_selected.iloc[row_index]

        # Extract values
        current_id = selected_row["IDs"]
        current_idx = selected_row["index"]
        pred_rthgb = selected_row["Pred_RTHGB"]
        pred_dw = selected_row["Pred_DW"]
        pred_hf = selected_row["Pred_HF"]
        pred_idh = selected_row["Pred_IDH_prob"]

        tsat = selected_row["TAST"]
        ferritin = selected_row["FERRITIN"]
        ferr_outlier = selected_row["ferr_outlier_H"]
        ferr_date = selected_row["PTIMESTAMP"].strftime("%Y-%m-%d")
        latest_esa = selected_row["LatestESA"]
        latest_esa_date = selected_row["RTHGB_DATE"].strftime("%Y-%m-%d")

        # Risk displays
        hf_text = "High Risk" if pred_hf == 1 else "Low Risk"
        hf_color = config.colors["high_risk"] if pred_hf == 1 else config.colors["low_risk"]
        hf_display = html.Span(hf_text, style={"color": hf_color})

        idh_text = "High Risk" if pred_idh >= config.idh_risk_threshold else "Low Risk"
        idh_color = config.colors["high_risk"] if pred_idh >= config.idh_risk_threshold else config.colors["low_risk"]
        idh_display = html.Span(idh_text, style={"color": idh_color})

        # Clinical recommendations
        if pred_hf == 1:
            doctor_txt = (
                "High risk of heart failure. Recommend reducing ultrafiltration rate "
                "and daily fluid intake. Adjust UF and dialysis parameters to reduce "
                "cardiac load. Consider inotropes or diuretics if needed. Monitor "
                "daily weight and cardiac function (BNP, echocardiogram)."
            )
            doctor_color = config.colors["high_risk"]
        elif pred_idh >= config.idh_risk_threshold:
            doctor_txt = (
                "High risk of intradialytic hypotension. Low venous pressure and "
                "unstable blood pressure detected. Recommend lowering dialysate "
                "temperature and reducing UF rate. Improve nutrition (protein, sodium). "
                "Monitor blood pressure dynamically during dialysis."
            )
            doctor_color = config.colors["high_risk"]
        else:
            doctor_txt = (
                "Low risk patient. All indicators stable. Continue current dialysis "
                "plan. Maintain stable blood pressure and fluid balance. Avoid high "
                "sodium diet, monitor fluid intake, and track RTHGB and iron indices."
            )
            doctor_color = config.colors["low_risk"]

        doctor_display = html.Span(doctor_txt, style={"color": doctor_color})

        # Text displays
        id_display = f"Patient ID: {current_id}"
        hgb_display = f"RTHGB (Hemoglobin): {pred_rthgb}"
        dw_display = f"DW (Dry Weight): {pred_dw} kg"

        ferr_display = f"Last FERRITIN:\n{ferr_outlier} mg"
        ferr_date_display = f"Last FERRITIN Date:\n{ferr_date}"
        esa_display = f"Last ESA:\n{latest_esa} U"
        esa_date_display = f"Last ESA Date:\n{latest_esa_date}"

        # Time series data
        df_patient = df[df["IDs"] == current_id]
        current_pos = df_patient[df_patient["index"] == current_idx].index[0]
        update_data = df_patient.loc[current_pos-10:current_pos].tail(config.time_series_window)

        # Create time series charts
        def create_line_chart(data, y_col, title):
            fig = go.Figure()
            x_vals = data["PTIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            y_vals = data[y_col].tolist()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=y_col))
            fig.update_layout(
                title=dict(text=title, font=dict(size=12)),
                template="plotly_dark",
                title_x=0.5,
                margin=dict(l=25, r=25, t=30, b=30),
                yaxis=dict(tickfont=dict(size=8)),
                xaxis=dict(tickfont=dict(size=8))
            )
            return fig

        fig_tmp = create_line_chart(update_data, "A_TMP", config.chart_titles["A_TMP"])
        fig_vp = create_line_chart(update_data, "A_VENOUSPRESSURE", config.chart_titles["A_VENOUSPRESSURE"])
        fig_ap = create_line_chart(update_data, "A_ARTERIALPRESSURE", config.chart_titles["A_ARTERIALPRESSURE"])
        fig_uf = create_line_chart(update_data, "A_TOTALUF", config.chart_titles["A_TOTALUF"])
        fig_temp = create_line_chart(update_data, "A_D_TEMPERATURE", config.chart_titles["A_D_TEMPERATURE"])
        fig_bicarb = create_line_chart(update_data, "A_BICARBONATEADJUSTMENT", config.chart_titles["A_BICARBONATEADJUSTMENT"])

        # Gauge charts
        fig_tsat = go.Figure(go.Indicator(
            mode="gauge+number",
            value=tsat,
            title={"text": "TSAT (Transferrin Saturation)", "font": {"size": 16, "color": "white"}},
            gauge={
                "axis": {"range": [0, tsat], "tickwidth": 1, "showticklabels": False},
                "bar": {"color": config.colors["tsat_gauge"]}
            },
            number={"suffix": " %"}
        ))
        fig_tsat.update_layout(
            paper_bgcolor=config.colors["background"],
            font={"size": 12, "color": "white"},
            margin=dict(l=70, r=70, t=70, b=70)
        )

        fig_ferritin = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ferritin,
            title={"text": "FERRITIN", "font": {"size": 16, "color": "white"}},
            gauge={
                "axis": {"range": [0, ferritin], "tickwidth": 1, "showticklabels": False},
                "bar": {"color": config.colors["ferritin_gauge"]}
            },
            number={"suffix": " ng/ml"}
        ))
        fig_ferritin.update_layout(
            paper_bgcolor=config.colors["background"],
            font={"size": 12, "color": "white"},
            margin=dict(l=70, r=70, t=70, b=70)
        )

        # Triangle chart
        fig_triangle = go.Figure()
        fig_triangle.add_trace(go.Scatterpolar(
            r=[pred_rthgb, ferritin, latest_esa, pred_rthgb],
            theta=["Hemoglobin", "Iron", "ESA", "Hemoglobin"],
            fill="toself",
            line=dict(color=config.colors["triangle"])
        ))
        fig_triangle.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    showticklabels=False,
                    range=[0, max(pred_rthgb, ferritin, latest_esa) + 1]
                )
            ),
            showlegend=False,
            template="plotly_dark",
            title_x=0.5,
            font={"size": 12, "color": "white"},
            title=dict(text="Hemoglobin/Iron/ESA Triangle", font=dict(size=16))
        )

        return (
            id_display, hf_display, hgb_display, dw_display, idh_display,
            fig_tsat, fig_ferritin, ferr_display, ferr_date_display,
            esa_display, esa_date_display, fig_triangle,
            fig_tmp, fig_vp, fig_ap, fig_uf, fig_temp, fig_bicarb,
            doctor_display
        )


def run_server(app: dash.Dash, config: Optional[DashboardConfig] = None):
    """
    Run the dashboard server.

    Args:
        app: Dash application instance
        config: Dashboard configuration
    """
    config = config or DashboardConfig()
    # Use app.run() for Dash 2.x+ (app.run_server is deprecated)
    app.run(host=config.host, port=config.port, debug=config.debug)


# Main entry point
if __name__ == "__main__":
    app = create_app()
    run_server(app)

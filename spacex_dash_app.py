#!/usr/bin/env python3
"""
IBM Applied Data Science Capstone - SpaceX Plotly Dash application.

This app CONSUMES spacex_launch_dash.csv. It does not generate the project
datasets; use generate_spacex_data.py for that.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


DATA_FILE = Path(__file__).with_name("spacex_launch_dash.csv")
spacex_df = pd.read_csv(DATA_FILE)

max_payload = float(spacex_df["Payload Mass (kg)"].max())
min_payload = float(spacex_df["Payload Mass (kg)"].min())

app = Dash(__name__)

site_options = [{"label": "All Sites", "value": "ALL"}]
site_options.extend(
    {"label": site, "value": site}
    for site in sorted(spacex_df["Launch Site"].dropna().unique())
)

app.layout = html.Div(
    [
        html.H1(
            "SpaceX Launch Records Dashboard",
            style={"textAlign": "center"},
        ),
        dcc.Dropdown(
            id="site-dropdown",
            options=site_options,
            value="ALL",
            placeholder="Select a Launch Site here",
            searchable=True,
        ),
        html.Br(),
        dcc.Graph(id="success-pie-chart"),
        html.P("Payload range (kg):"),
        dcc.RangeSlider(
            id="payload-slider",
            min=0,
            max=10000,
            step=1000,
            marks={i: str(i) for i in range(0, 10001, 2500)},
            value=[min_payload, max_payload],
        ),
        dcc.Graph(id="success-payload-scatter-chart"),
    ]
)


@app.callback(
    Output("success-pie-chart", "figure"),
    Input("site-dropdown", "value"),
)
def render_success_pie(selected_site):
    if selected_site == "ALL":
        success_by_site = (
            spacex_df.groupby("Launch Site", as_index=False)["class"].sum()
        )
        return px.pie(
            success_by_site,
            values="class",
            names="Launch Site",
            title="Total Successful Launches by Site",
        )

    filtered = spacex_df[spacex_df["Launch Site"] == selected_site]
    outcome_counts = (
        filtered["class"]
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
        .sort_values("class")
    )
    outcome_counts["Outcome"] = outcome_counts["class"].map(
        {0: "Failure", 1: "Success"}
    )
    return px.pie(
        outcome_counts,
        values="count",
        names="Outcome",
        title=f"Launch Outcomes for {selected_site}",
    )


@app.callback(
    Output("success-payload-scatter-chart", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def render_payload_scatter(selected_site, payload_range):
    low, high = payload_range

    filtered = spacex_df[
        spacex_df["Payload Mass (kg)"].between(low, high, inclusive="both")
    ]

    if selected_site != "ALL":
        filtered = filtered[filtered["Launch Site"] == selected_site]
        title = f"Payload vs. Outcome for {selected_site}"
    else:
        title = "Payload vs. Outcome for All Sites"

    return px.scatter(
        filtered,
        x="Payload Mass (kg)",
        y="class",
        color="Booster Version Category",
        hover_data=["Launch Site", "Flight Number", "Booster Version"],
        title=title,
        labels={"class": "Landing Success (0 = Failure, 1 = Success)"},
    )


if __name__ == "__main__":
    app.run(debug=False, port=8050)

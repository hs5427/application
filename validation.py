import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_sample_quantile_table(
    df_sample,
    quantile_probs,
    q_col="TTP_Percentile",
    value_col="value",
    n_bins=8
):
    df = df_sample[[q_col, value_col]].dropna().copy()

    q_min = df[q_col].min()
    q_max = df[q_col].max()

    bins = np.linspace(q_min, q_max, n_bins + 1)

    df["TTP_bin"] = pd.cut(
        df[q_col],
        bins=bins,
        include_lowest=True
    )

    rows = []

    for interval, df_group in df.groupby("TTP_bin", observed=True):
        if df_group.empty:
            continue

        q_center = (interval.left + interval.right) / 2

        for p in quantile_probs:
            sample_value = df_group[value_col].quantile(p)

            rows.append({
                "TTP_Percentile": q_center,
                "value_Percentile": p,
                "sample_value": sample_value
            })

    return pd.DataFrame(rows)


def make_validation_table(
    df_long,
    df_sample,
    q_col="TTP_Percentile",
    p_col="value_Percentile",
    value_col="value",
    n_bins=8
):
    quantile_probs = sorted(df_long[p_col].dropna().unique())

    df_sample_q = make_sample_quantile_table(
        df_sample=df_sample,
        quantile_probs=quantile_probs,
        q_col=q_col,
        value_col=value_col,
        n_bins=n_bins
    )

    df_actual = df_long[[q_col, p_col, value_col]].copy()
    df_actual = df_actual.rename(columns={
        value_col: "actual_value"
    })

    df_actual = df_actual.sort_values(q_col)

    merged_rows = []

    for _, row in df_sample_q.iterrows():
        q = row[q_col]
        p = row[p_col]

        df_same_p = df_actual[df_actual[p_col] == p].copy()

        if df_same_p.empty:
            continue

        idx = (df_same_p[q_col] - q).abs().idxmin()
        actual_value = df_same_p.loc[idx, "actual_value"]

        merged_rows.append({
            q_col: q,
            p_col: p,
            "actual_value": actual_value,
            "sample_value": row["sample_value"],
            "error": row["sample_value"] - actual_value,
            "abs_error": abs(row["sample_value"] - actual_value)
        })

    df_validation = pd.DataFrame(merged_rows)

    return df_validation


def make_validation_metrics(df_validation):
    if df_validation is None or df_validation.empty:
        return pd.DataFrame()

    rmse = np.sqrt(np.mean(df_validation["error"] ** 2))
    mae = np.mean(df_validation["abs_error"])
    max_abs_error = np.max(df_validation["abs_error"])

    return pd.DataFrame({
        "metric": ["RMSE", "MAE", "Max Abs Error"],
        "value": [rmse, mae, max_abs_error]
    })


def plot_validation_diagnostics(
    df_long,
    df_sample,
    df_validation,
    selected_feature="value",
    q_col="TTP_Percentile",
    p_col="value_Percentile",
    value_col="value"
):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Original CDF",
            "Generated samples",
            "Q-Q plot"
        ]
    )

    # 1. Original CDF
    for q, df_group in df_long.groupby(q_col):
        df_group = df_group.sort_values(value_col)

        fig.add_trace(
            go.Scatter(
                x=df_group[value_col],
                y=df_group[p_col],
                mode="lines+markers",
                name=f"Original TTP={q}",
                legendgroup=f"TTP={q}",
                showlegend=False
            ),
            row=1,
            col=1
        )

    # 2. Generated scatter
    fig.add_trace(
        go.Scatter(
            x=df_sample["TTP"],
            y=df_sample[value_col],
            mode="markers",
            marker=dict(size=4, opacity=0.35),
            name="Generated samples",
            showlegend=False
        ),
        row=1,
        col=2
    )

    # 3. Q-Q plot
    if df_validation is not None and not df_validation.empty:
        fig.add_trace(
            go.Scatter(
                x=df_validation["actual_value"],
                y=df_validation["sample_value"],
                mode="markers",
                marker=dict(size=7, opacity=0.7),
                name="Q-Q",
                showlegend=False
            ),
            row=1,
            col=3
        )

        max_val = max(
            df_validation["actual_value"].max(),
            df_validation["sample_value"].max()
        )

        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(dash="dash"),
                name="45 degree line",
                showlegend=False
            ),
            row=1,
            col=3
        )

        fig.update_xaxes(range=[0, max_val * 1.05], row=1, col=3)
        fig.update_yaxes(range=[0, max_val * 1.05], row=1, col=3)

    fig.update_xaxes(title_text=selected_feature, row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=1, col=1)

    fig.update_xaxes(title_text="TTP", row=1, col=2)
    fig.update_yaxes(title_text=selected_feature, row=1, col=2)

    fig.update_xaxes(title_text=f"Original {selected_feature}", row=1, col=3)
    fig.update_yaxes(title_text=f"Generated {selected_feature}", row=1, col=3)

    fig.update_layout(
        height=650,
        width=1800,
        title="Validation diagnostics"
    )

    return fig
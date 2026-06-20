import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def make_qq_table_by_ttp(
    df_long,
    df_sample,
    percentiles=None,
    q_col="TTP_Percentile",
    p_col="value_Percentile",
    value_col="value",
    ttp_bin_width=2.5
):
    if percentiles is None:
        percentiles = [
            0.01, 0.25, 0.50, 0.75,
            0.90, 0.95, 0.99, 0.998
        ]

    rows = []

    ttp_values = sorted(df_long[q_col].dropna().unique())

    for ttp in ttp_values:

        df_org_ttp = df_long[
            np.isclose(df_long[q_col], ttp)
        ].copy()

        # ここが重要：完全一致ではなく近傍サンプルを使う
        df_gen_ttp = df_sample[
            (df_sample[q_col] >= ttp - ttp_bin_width) &
            (df_sample[q_col] <= ttp + ttp_bin_width)
        ].copy()

        if df_org_ttp.empty or df_gen_ttp.empty:
            continue

        for p in percentiles:

            df_org_p = df_org_ttp[
                np.isclose(df_org_ttp[p_col], p)
            ]

            if df_org_p.empty:
                continue

            original_value = float(
                df_org_p[value_col].iloc[0]
            )

            generated_value = float(
                np.percentile(
                    df_gen_ttp[value_col].dropna().to_numpy(dtype=float),
                    p * 100
                )
            )

            rows.append({
                q_col: ttp,
                p_col: p,
                "Original": original_value,
                "Generated": generated_value,
                "SampleCount": len(df_gen_ttp),
                "Error": generated_value - original_value,
                "AbsError": abs(generated_value - original_value)
            })

    return pd.DataFrame(rows)


def plot_qq_by_percentile_grid(
    df_qq,
    selected_feature="value",
    percentiles=None,
    p_col="value_Percentile",
    q_col="TTP_Percentile"
):
    """
    PercentileごとにQ-Q plotを分けて、4列×2段で描画する。
    各subplotでは、各TTP_Percentileが1点になる。
    """

    if df_qq is None or df_qq.empty:
        return None

    if percentiles is None:
        percentiles = [
            0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.998
        ]

    subplot_titles = [
        f"{p}%" for p in percentiles
    ]

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.25
    )

    fig.update_layout(
        height=2500
    )

    max_val = max(
        df_qq["Original"].max(),
        df_qq["Generated"].max()
    )

    for i, p in enumerate(percentiles):

        row = i // 4 + 1
        col = i % 4 + 1

        df_p = df_qq[
            np.isclose(df_qq[p_col], p)
        ].copy()

        if df_p.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_p["Original"],
                y=df_p["Generated"],
                mode="markers",
                marker=dict(size=7, opacity=0.75),
                text=df_p[q_col],
                customdata=df_p[[q_col, p_col, "Error"]],
                hovertemplate=(
                    "TTP=%{customdata[0]}<br>"
                    "Percentile=%{customdata[1]}<br>"
                    "Original=%{x}<br>"
                    "Generated=%{y}<br>"
                    "Error=%{customdata[2]}<extra></extra>"
                ),
                name=f"{p}%",
                showlegend=False
            ),
            row=row,
            col=col
        )

        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(dash="dash"),
                showlegend=False
            ),
            row=row,
            col=col
        )

        if row == 2:
            x_title = "Original"
        else:
            x_title = None

        fig.update_xaxes(
            title_text=x_title,
            range=[0, max_val * 1.05],
            row=row,
            col=col
        )

        fig.update_yaxes(
            title_text="Generated",
            range=[0, max_val * 1.05],
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=None)
        fig.update_yaxes(title_text=None)

    fig.update_layout(
        title=dict(
            text=f"Q-Q plot by TTP - {selected_feature}",
            x=0.5,
            y=0.98
        ),
        height=1000,
        margin=dict(
            t=120,
            b=50,
            l=50,
            r=50
        ),
        showlegend=False
    )

    fig.update_layout(
        annotations=[
            dict(
                text="Original",
                x=0.5,
                y=-0.05,
                showarrow=False
            ),
            dict(
                text="Generated",
                x=-0.05,
                y=0.5,
                textangle=-90,
                showarrow=False
            )
        ]
    )

    return fig


def plot_qq_by_ttp_grid(
    df_qq,
    selected_feature="value",
    q_col="TTP_Percentile",
    p_col="value_Percentile",
    n_cols=4
):

    if df_qq is None or df_qq.empty:
        return None

    ttp_values = sorted(df_qq[q_col].dropna().unique())

    n_plots = len(ttp_values)
    n_rows = math.ceil(n_plots / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"TTP={ttp:g}" for ttp in ttp_values],
        horizontal_spacing=0.06,
        vertical_spacing=0.25
    )

    max_val = max(
        df_qq["Original"].max(),
        df_qq["Generated"].max()
    )

    min_val = min(
        df_qq["Original"].min(),
        df_qq["Generated"].min()
    )

    axis_min = min(0, min_val)
    axis_max = max_val * 1.05

    for i, ttp in enumerate(ttp_values):
        row = i // n_cols + 1
        col = i % n_cols + 1

        df_ttp = df_qq[np.isclose(df_qq[q_col], ttp)].copy()

        if df_ttp.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_ttp["Original"],
                y=df_ttp["Generated"],
                mode="markers+text",
                text=(df_ttp[p_col] * 100).map(lambda x: f"{x:g}%"),
                textposition="top center",
                marker=dict(size=8, opacity=0.8),
                customdata=df_ttp[[q_col, p_col, "Error"]],
                hovertemplate=(
                    "TTP=%{customdata[0]}<br>"
                    "Percentile=%{customdata[1]}<br>"
                    "Original=%{x}<br>"
                    "Generated=%{y}<br>"
                    "Error=%{customdata[2]}<extra></extra>"
                ),
                showlegend=False
            ),
            row=row,
            col=col
        )

        fig.add_trace(
            go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                line=dict(dash="dash"),
                showlegend=False
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(
            title_text="Original",
            range=[axis_min, axis_max],
            showgrid=True,
            zeroline=True,
            row=row,
            col=col
        )

        fig.update_yaxes(
            title_text="Generated",
            range=[axis_min, axis_max],
            showgrid=True,
            zeroline=True,
            scaleanchor=f"x{i+1}" if i > 0 else "x",
            scaleratio=1,
            row=row,
            col=col
        )

    fig.update_layout(
        height=320 * n_rows,
        title=f"Q-Q plot by TTP Percentile - {selected_feature}",
        showlegend=False
    )

    return fig


def make_independent_qq_table(
    df_long,
    df_sample,
    percentiles=None,
    p_col="value_Percentile",
    value_col="value"
):

    if percentiles is None:
        percentiles = [
            0.01, 0.25, 0.50, 0.75,
            0.90, 0.95, 0.99, 0.998
        ]

    df_ref = df_long[[p_col, value_col]].dropna().copy()
    df_ref[p_col] = df_ref[p_col].astype(float)
    df_ref[value_col] = df_ref[value_col].astype(float)

    sample_values = (
        df_sample[value_col]
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(sample_values) == 0:
        raise ValueError("df_sampleに有効なvalueがありません。")

    rows = []

    for p in percentiles:

        df_p = df_ref[
            np.isclose(df_ref[p_col], p)
        ]

        if df_p.empty:
            original = np.nan
        else:
            original = float(df_p[value_col].iloc[0])

        generated = float(
            np.percentile(
                sample_values,
                p * 100
            )
        )

        rows.append({
            "Percentile": p,
            "Original": original,
            "Generated": generated,
            "Error": generated - original if not np.isnan(original) else np.nan,
            "AbsError": abs(generated - original) if not np.isnan(original) else np.nan
        })

    df_qq = pd.DataFrame(rows)

    return df_qq


def plot_independent_qq(
    df_qq,
    selected_feature="value"
):

    if df_qq is None or df_qq.empty:
        return None

    required_cols = ["Original", "Generated", "Percentile"]

    for col in required_cols:
        if col not in df_qq.columns:
            raise ValueError(
                f"df_qqに {col} 列がありません。現在の列: {df_qq.columns.tolist()}"
            )

    df_plot = df_qq.dropna(
        subset=["Original", "Generated"]
    ).copy()

    if df_plot.empty:
        raise ValueError(
            "Original / Generated がすべてNaNです。Q-Q plotに描画する点がありません。"
        )

    min_val = min(
        df_plot["Original"].min(),
        df_plot["Generated"].min()
    )

    max_val = max(
        df_plot["Original"].max(),
        df_plot["Generated"].max()
    )

    axis_min = min(0, min_val)
    axis_max = max_val * 1.05

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_plot["Original"],
            y=df_plot["Generated"],
            mode="markers+text",
            text=(df_plot["Percentile"] * 100).map(lambda x: f"{x:g}%"),
            textposition="top center",
            marker=dict(size=10, opacity=0.8),
            customdata=df_plot[["Percentile", "Error"]],
            hovertemplate=(
                "Percentile=%{customdata[0]}<br>"
                "Original=%{x}<br>"
                "Generated=%{y}<br>"
                "Error=%{customdata[1]}<extra></extra>"
            ),
            name="Q-Q points"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line=dict(dash="dash"),
            name="45 degree line"
        )
    )

    fig.update_xaxes(
        title_text="Original",
        range=[axis_min, axis_max],
        showgrid=True,
        zeroline=True
    )

    fig.update_yaxes(
        title_text="Generated",
        range=[axis_min, axis_max],
        showgrid=True,
        zeroline=True,
        scaleanchor="x",
        scaleratio=1
    )

    fig.update_layout(
        title=f"Independent Q-Q plot - {selected_feature}",
        height=700,
        showlegend=False
    )

    return fig


def make_validation_metrics_from_qq(df_qq):
    import numpy as np
    import pandas as pd

    if df_qq is None or df_qq.empty:
        return pd.DataFrame()

    required_cols = ["Original", "Generated"]

    for col in required_cols:
        if col not in df_qq.columns:
            raise ValueError(
                f"df_qqに {col} 列がありません。現在の列: {df_qq.columns.tolist()}"
            )

    df = df_qq[["Original", "Generated"]].dropna().copy()

    if df.empty:
        return pd.DataFrame()

    y_true = df["Original"].to_numpy(dtype=float)
    y_pred = df["Generated"].to_numpy(dtype=float)

    # =========================
    # Error metrics
    # =========================

    error = y_pred - y_true

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mae = np.mean(
        np.abs(error)
    )

    max_abs_error = np.max(
        np.abs(error)
    )

    mean_error = np.mean(error)

    # =========================
    # QQ correlation R2
    # =========================
    # 相関係数^2。
    # 0〜1の範囲になり、QQプロットの直線性を表す。
    # ただし、傾き1・切片0かどうかは別途 slope/intercept で見る。

    if len(y_true) < 2:
        qq_corr = np.nan
        qq_r2 = np.nan
    elif np.std(y_true) == 0 or np.std(y_pred) == 0:
        qq_corr = np.nan
        qq_r2 = np.nan
    else:
        qq_corr = np.corrcoef(
            y_true,
            y_pred
        )[0, 1]

        qq_r2 = qq_corr ** 2

    # =========================
    # QQ regression line
    # =========================
    # Generated = slope * Original + intercept
    #
    # 理想:
    #   slope     = 1
    #   intercept = 0

    if len(y_true) < 2 or np.std(y_true) == 0:
        slope = np.nan
        intercept = np.nan
    else:
        slope, intercept = np.polyfit(
            y_true,
            y_pred,
            deg=1
        )

    # =========================
    # Legacy R2: can be negative
    # =========================
    # 参考用。平均予測より悪いとマイナスになる。

    sse = np.sum(
        (y_true - y_pred) ** 2
    )

    sst = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if sst == 0:
        legacy_r2 = np.nan
    else:
        legacy_r2 = 1 - sse / sst

    df_metrics = pd.DataFrame({
        "Metric": [
            "QQ_R2_corr",
            "QQ_corr",
            "QQ_slope",
            "QQ_intercept",
            "RMSE",
            "MAE",
            "MaxAbsError",
            "MeanError",
            "Legacy_R2"
        ],
        "Value": [
            qq_r2,
            qq_corr,
            slope,
            intercept,
            rmse,
            mae,
            max_abs_error,
            mean_error,
            legacy_r2
        ]
    })

    return df_metrics

# def make_validation_metrics_from_qq(df_qq):

#     if df_qq is None or df_qq.empty:
#         return pd.DataFrame()

#     df = df_qq[["Original", "Generated"]].dropna().copy()

#     if df.empty:
#         return pd.DataFrame()

#     y_true = df["Original"].to_numpy(dtype=float)
#     y_pred = df["Generated"].to_numpy(dtype=float)

#     sse = np.sum((y_true - y_pred) ** 2)
#     sst = np.sum((y_true - np.mean(y_true)) ** 2)

#     if sst == 0:
#         r2 = np.nan
#     else:
#         r2 = 1 - sse / sst

#     rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
#     mae = np.mean(np.abs(y_true - y_pred))

#     df_metrics = pd.DataFrame({
#         "Metric": [
#             "R2",
#             "RMSE",
#             "MAE"
#         ],
#         "Value": [
#             r2,
#             rmse,
#             mae
#         ]
#     })

#     return df_metrics


def plot_sample_distribution_overlay(
    df_sample,
    df_reference,
    selected_feature="value",
    value_col="value",
    p_col="value_Percentile",
    q_col="TTP_Percentile",
    analysis_mode="Independent",
    target_ttp=None,
    bins=50
):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde

    if df_sample is None or df_sample.empty:
        return None

    sample_values = df_sample[value_col].dropna().to_numpy(dtype=float)

    if len(sample_values) < 2:
        return None

    x_min = float(np.min(sample_values))
    x_max = float(np.max(sample_values))

    x_grid = np.linspace(x_min, x_max, 500)

    # =========================
    # PDF: KDE
    # =========================

    kde = gaussian_kde(sample_values)
    pdf_values = kde(x_grid)

    # =========================
    # CDF: empirical CDF
    # =========================

    sorted_sample = np.sort(sample_values)
    cdf_values = np.searchsorted(
        sorted_sample,
        x_grid,
        side="right"
    ) / len(sorted_sample)

    fig = go.Figure()

    # =========================
    # Histogram
    # =========================

    fig.add_trace(
        go.Histogram(
            x=sample_values,
            nbinsx=bins,
            histnorm="probability density",
            opacity=0.35,
            name="Sample histogram",
            yaxis="y1"
        )
    )

    # =========================
    # PDF
    # =========================

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=pdf_values,
            mode="lines",
            name="Sample PDF",
            yaxis="y1"
        )
    )

    # =========================
    # CDF
    # =========================

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=cdf_values,
            mode="lines",
            name="Sample CDF",
            yaxis="y2"
        )
    )

    # =========================
    # Reference measured points
    # =========================

    df_ref = df_reference.copy()

    if analysis_mode == "TTP dependent":
        if target_ttp is None:
            target_ttp = df_ref[q_col].dropna().unique()[0]

        df_ref = df_ref[
            np.isclose(df_ref[q_col].astype(float), float(target_ttp))
        ].copy()

    df_ref = df_ref[[p_col, value_col]].dropna().copy()

    if not df_ref.empty:
        df_ref[p_col] = df_ref[p_col].astype(float)
        df_ref[value_col] = df_ref[value_col].astype(float)

        fig.add_trace(
            go.Scatter(
                x=df_ref[value_col],
                y=df_ref[p_col],
                mode="markers+text",
                text=(df_ref[p_col] * 100).map(lambda x: f"{x:g}%"),
                textposition="top center",
                name="Measured CDF points",
                yaxis="y2"
            )
        )
    
    fig.update_layout(
        height=650,
        title="Sample distribution overlay",
        xaxis=dict(
            title=selected_feature,
            showgrid=True
        ),
        yaxis=dict(
            title="PDF / density",
            side="left",
            showgrid=True,
            zeroline=True
        ),
        yaxis2=dict(
            title="CDF / cumulative probability",
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False,
            zeroline=False
        ),
        bargap=0.02,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_sample_pdf_histogram(
    df_sample,
    selected_feature="value",
    value_col="value",
    bins=50
):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde

    if df_sample is None or df_sample.empty:
        return None

    x = df_sample[value_col].dropna().to_numpy(dtype=float)

    if len(x) < 2:
        return None

    x_min = float(np.min(x))
    x_max = float(np.max(x))

    x_grid = np.linspace(x_min, x_max, 500)

    kde = gaussian_kde(x)
    pdf = kde(x_grid)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=x,
            nbinsx=bins,
            histnorm="probability density",
            opacity=0.35,
            name="Sample histogram"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=pdf,
            mode="lines",
            name="Sample PDF"
        )
    )

    fig.update_layout(
        height=600,
        title="Sample PDF + Histogram",
        xaxis_title=selected_feature,
        yaxis_title="Density",
        bargap=0.02,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_cdf_validation(
    df_sample,
    df_reference,
    selected_feature="value",
    value_col="value",
    p_col="value_Percentile",
    q_col="TTP_Percentile",
    analysis_mode="Independent",
    target_ttp=None
):

    if df_sample is None or df_sample.empty:
        return None

    if analysis_mode == "TTP dependent":
        if target_ttp is None:
            return None

        df_sample_plot = df_sample[
            np.isclose(
                df_sample[q_col].astype(float),
                float(target_ttp)
            )
        ].copy()

        df_ref_plot = df_reference[
            np.isclose(
                df_reference[q_col].astype(float),
                float(target_ttp)
            )
        ].copy()

    else:
        df_sample_plot = df_sample.copy()
        df_ref_plot = df_reference.copy()

    sample_values = df_sample_plot[value_col].dropna().to_numpy(dtype=float)

    if len(sample_values) < 2:
        return None

    x_sorted = np.sort(sample_values)
    y_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=y_cdf,
            mode="lines",
            name="Generated CDF"
        )
    )

    if p_col in df_ref_plot.columns and value_col in df_ref_plot.columns:
        df_ref_plot = df_ref_plot[[p_col, value_col]].dropna().copy()
        df_ref_plot[p_col] = df_ref_plot[p_col].astype(float)
        df_ref_plot[value_col] = df_ref_plot[value_col].astype(float)

        fig.add_trace(
            go.Scatter(
                x=df_ref_plot[value_col],
                y=df_ref_plot[p_col],
                mode="markers+text",
                text=(df_ref_plot[p_col] * 100).map(lambda x: f"{x:g}%"),
                textposition="top center",
                name="Original percentile points"
            )
        )

    title = "CDF validation"

    if analysis_mode == "TTP dependent":
        title += f" - TTP={target_ttp:g}"

    fig.update_layout(
        height=800,
        title=title,
        xaxis_title=selected_feature,
        yaxis_title="Cumulative probability",
        yaxis=dict(
            range=[0, 1],
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_cdf_validation_grid(
    df_sample,
    df_reference,
    selected_feature="value",
    value_col="value",
    p_col="value_Percentile",
    q_col="TTP_Percentile",
    ttp_bin_width=0.025
):

    if df_sample is None or df_sample.empty:
        return None

    if q_col not in df_reference.columns:
        return None

    if q_col not in df_sample.columns:
        raise ValueError(
            f"df_sampleに {q_col} 列がありません。"
            f"現在の列: {df_sample.columns.tolist()}"
        )

    ttp_values = sorted(
        df_reference[q_col].dropna().astype(float).unique()
    )

    n_plots = min(len(ttp_values), 8)

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[
            f"TTP={float(v):g}"
            for v in ttp_values[:n_plots]
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.08
    )

    for i, ttp in enumerate(ttp_values[:n_plots]):

        row = i // 4 + 1
        col = i % 4 + 1

        # =========================
        # Generated sample CDF
        # =========================
        # サンプル側のTTP_Percentileは連続値の可能性があるため、
        # 実測TTP点の周辺binで抽出する。

        lower = float(ttp) - ttp_bin_width / 2
        upper = float(ttp) + ttp_bin_width / 2

        df_s = df_sample[
            (df_sample[q_col].astype(float) >= lower)
            & (df_sample[q_col].astype(float) < upper)
        ].copy()

        if not df_s.empty:
            sample_values = (
                df_s[value_col]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(sample_values) >= 2:
                x_sorted = np.sort(sample_values)
                y_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

                fig.add_trace(
                    go.Scatter(
                        x=x_sorted,
                        y=y_cdf,
                        mode="lines",
                        name="Generated CDF",
                        line=dict(width=3),
                        showlegend=(i == 0)
                    ),
                    row=row,
                    col=col
                )

        # =========================
        # Original CDF
        # =========================

        df_r = df_reference[
            np.isclose(
                df_reference[q_col].astype(float),
                float(ttp)
            )
        ].copy()

        if not df_r.empty:
            df_r = df_r[[p_col, value_col]].dropna().copy()
            df_r[p_col] = df_r[p_col].astype(float)
            df_r[value_col] = df_r[value_col].astype(float)
            df_r = df_r.sort_values(value_col)

            fig.add_trace(
                go.Scatter(
                    x=df_r[value_col],
                    y=df_r[p_col],
                    mode="markers",
                    marker=dict(size=8),
                    line=dict(dash="dash"),
                    name="Original CDF",
                    showlegend=(i == 0)
                ),
                row=row,
                col=col
            )

        fig.update_yaxes(
            range=[0, 1],
            title_text="CDF" if col == 1 else None,
            row=row,
            col=col
        )

        fig.update_xaxes(
            title_text=selected_feature if row == 2 else None,
            row=row,
            col=col
        )

    fig.update_layout(
        height=800,
        title="CDF validation by TTP Percentile",
        margin=dict(t=120, b=80, l=60, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1
        )
    )

    return fig
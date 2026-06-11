import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        subplot_titles=subplot_titles
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

        fig.update_xaxes(
            title_text="Original",
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

    fig.update_layout(
        height=900,
        title=f"Q-Q plot by percentile - {selected_feature}",
        showlegend=False
    )

    return fig


# def plot_qq_by_ttp_grid(
#     df_qq,
#     selected_feature="value",
#     q_col="TTP_Percentile",
#     p_col="value_Percentile",
#     n_cols=4
# ):
#     import math
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots

#     if df_qq is None or df_qq.empty:
#         return None

#     ttp_values = sorted(df_qq[q_col].dropna().unique())

#     n_plots = len(ttp_values)
#     n_rows = math.ceil(n_plots / n_cols)

#     subplot_titles = [
#         f"TTP={ttp:g}" for ttp in ttp_values
#     ]

#     fig = make_subplots(
#         rows=n_rows,
#         cols=n_cols,
#         subplot_titles=subplot_titles
#     )

#     max_val = max(
#         df_qq["Original"].max(),
#         df_qq["Generated"].max()
#     )

#     for i, ttp in enumerate(ttp_values):
#         row = i // n_cols + 1
#         col = i % n_cols + 1

#         df_ttp = df_qq[df_qq[q_col] == ttp].copy()

#         if df_ttp.empty:
#             continue

#         fig.add_trace(
#             go.Scatter(
#                 x=df_ttp["Original"],
#                 y=df_ttp["Generated"],
#                 mode="markers+text",
#                 text=(df_ttp[p_col] * 100).map(lambda x: f"{x:g}%"),
#                 textposition="top center",
#                 marker=dict(size=8, opacity=0.75),
#                 customdata=df_ttp[[q_col, p_col, "Error"]],
#                 hovertemplate=(
#                     "TTP=%{customdata[0]}<br>"
#                     "Percentile=%{customdata[1]}<br>"
#                     "Original=%{x}<br>"
#                     "Generated=%{y}<br>"
#                     "Error=%{customdata[2]}<extra></extra>"
#                 ),
#                 name=f"TTP={ttp:g}",
#                 showlegend=False
#             ),
#             row=row,
#             col=col
#         )

#         fig.add_trace(
#             go.Scatter(
#                 x=[0, max_val],
#                 y=[0, max_val],
#                 mode="lines",
#                 line=dict(dash="dash"),
#                 showlegend=False
#             ),
#             row=row,
#             col=col
#         )

#         fig.update_xaxes(
#             title_text="Original",
#             range=[0, max_val * 1.05],
#             row=row,
#             col=col
#         )

#         fig.update_yaxes(
#             title_text="Generated",
#             range=[0, max_val * 1.05],
#             row=row,
#             col=col
#         )

#     fig.update_layout(
#         height=300 * n_rows,
#         title=f"Q-Q plot by TTP Percentile - {selected_feature}",
#         showlegend=False
#     )

#     return fig


def plot_qq_by_ttp_grid(
    df_qq,
    selected_feature="value",
    q_col="TTP_Percentile",
    p_col="value_Percentile",
    n_cols=4
):
    import math
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
        vertical_spacing=0.10
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
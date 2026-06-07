import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import least_squares
import plotly.graph_objects as go
from scipy.optimize import least_squares
from plotly.subplots import make_subplots
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

# =========================
# Weibull CDF
# =========================

def weibull_cdf(x, eta, beta):
    x = np.asarray(x, dtype=float)
    return 1 - np.exp(-(x / eta) ** beta)


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


# =========================
# K成分混合ワイブルCDF
# =========================

def mixture_weibull_cdf(x, params, n_components):
    """
    params:
        [log_eta_1, log_beta_1, ..., log_eta_K, log_beta_K, z_1, ..., z_K]
    """

    x = np.asarray(x, dtype=float)

    log_eta = params[0 : 2 * n_components : 2]
    log_beta = params[1 : 2 * n_components : 2]
    z_weight = params[2 * n_components : 3 * n_components]

    eta = np.exp(log_eta)
    beta = np.exp(log_beta)
    weights = softmax(z_weight)

    F = np.zeros_like(x, dtype=float)

    for k in range(n_components):
        F += weights[k] * weibull_cdf(x, eta[k], beta[k])

    return F


def residuals(params, x, p_obs, n_components):
    p_pred = mixture_weibull_cdf(x, params, n_components)
    return p_obs - p_pred


# =========================
# 初期値作成
# =========================

def make_initial_params(x, n_components):
    """
    etaはxの分位点でばらす。
    betaはすべて2.0。
    weightは均等。
    """

    quantiles = np.linspace(20, 80, n_components)

    eta_init = np.percentile(x, quantiles)
    beta_init = np.full(n_components, 2.0)

    log_eta_init = np.log(eta_init)
    log_beta_init = np.log(beta_init)

    z_weight_init = np.zeros(n_components)

    params = []

    for k in range(n_components):
        params.append(log_eta_init[k])
        params.append(log_beta_init[k])

    params.extend(z_weight_init)

    return np.array(params, dtype=float)


# =========================
# bounds作成
# =========================

def make_bounds(x, n_components):

    lower = []
    upper = []

    x_min = max(np.nanmin(x), 1e-12)
    x_max = max(np.nanmax(x), x_min * 10)

    for _ in range(n_components):
        # eta
        lower.append(np.log(x_min * 0.1))
        upper.append(np.log(x_max * 10))

        # beta
        lower.append(np.log(0.1))
        upper.append(np.log(100))

    # mixture weight用のz
    for _ in range(n_components):
        lower.append(-20)
        upper.append(20)

    return np.asarray(lower), np.asarray(upper)


# =========================
# 1グループをK成分混合ワイブルでフィット
# =========================

def fit_mixture_weibull(df_group, n_components=2):

    df_group = df_group.sort_values("value").copy()

    x = df_group["value"].to_numpy(dtype=float)
    p_obs = df_group["value_Percentile"].to_numpy(dtype=float)

    # ワイブルなのでx > 0のみ使用
    mask = x > 0
    x = x[mask]
    p_obs = p_obs[mask]

    # 0,1は不安定なので丸める
    p_obs = np.clip(p_obs, 1e-6, 1 - 1e-6)

    if len(x) < 2 * n_components + n_components:
        return None

    init_params = make_initial_params(x, n_components)
    bounds = make_bounds(x, n_components)

    result = least_squares(
        residuals,
        x0=init_params,
        args=(x, p_obs, n_components),
        bounds=bounds,
        max_nfev=20000
    )

    params_hat = result.x

    log_eta = params_hat[0 : 2 * n_components : 2]
    log_beta = params_hat[1 : 2 * n_components : 2]
    z_weight = params_hat[2 * n_components : 3 * n_components]

    eta = np.exp(log_eta)
    beta = np.exp(log_beta)
    weights = softmax(z_weight)

    rmse = np.sqrt(np.mean(result.fun ** 2))

    output = {
        "n_components": n_components,
        "rmse": rmse,
        # "success": result.success,
        # "message": result.message,
        # "params": params_hat
    }

    for k in range(n_components):
        output[f"eta{k+1}"] = eta[k]
        output[f"beta{k+1}"] = beta[k]
        output[f"w{k+1}"] = weights[k]

    return output

# =========================
# ThroughputPercentileごとにフィット
# =========================

def fit_by_throughput_percentile(df_long, n_components=2):
    fit_results = []

    for tp, df_group in df_long.groupby("TTP_Percentile"):
        try:
            fit = fit_mixture_weibull(
                df_group,
                n_components=n_components
            )

            if fit is None:
                continue

            fit["TTP_Percentile"] = tp
            fit_results.append(fit)

        except Exception:
            continue

    return pd.DataFrame(fit_results)

# =========================
# Global Model
# =========================

def build_global_mixture_weibull_cdf(
    df_fit_results_dict,
    n_components=2,
    q_col="TTP_Percentile"
):
    """
    df_fit_results_dict から、任意の ThroughputPercentile q に対する
    混合ワイブルCDF関数を作成する。

    Parameters
    ----------
    df_fit_results_dict : dict
        {1: df_fit_1, 2: df_fit_2, 3: df_fit_3} のような辞書

    n_components : int
        使用する混合ワイブルの成分数

    q_col : str
        ThroughputPercentile列名

    Returns
    -------
    global_mixture_weibull_cdf : function
        global_mixture_weibull_cdf(x, q)
    """

    df_fit = df_fit_results_dict.get(n_components)

    if df_fit is None or df_fit.empty:
        raise ValueError(f"{n_components}成分のフィット結果がありません。")

    df_fit = df_fit.sort_values(q_col).copy()

    q_values = df_fit[q_col].to_numpy(dtype=float)

    eta_splines = []
    beta_splines = []
    weight_splines = []

    for k in range(1, n_components + 1):
        eta_col = f"eta{k}"
        beta_col = f"beta{k}"
        w_col = f"w{k}"

        if eta_col not in df_fit.columns:
            raise ValueError(f"{eta_col} が df_fit にありません。")
        if beta_col not in df_fit.columns:
            raise ValueError(f"{beta_col} が df_fit にありません。")
        if w_col not in df_fit.columns:
            raise ValueError(f"{w_col} が df_fit にありません。")

        eta_values = df_fit[eta_col].to_numpy(dtype=float)
        beta_values = df_fit[beta_col].to_numpy(dtype=float)
        w_values = df_fit[w_col].to_numpy(dtype=float)

        # 正値制約のため log で補間
        eta_splines.append(
            CubicSpline(q_values, np.log(eta_values), extrapolate=True)
        )

        beta_splines.append(
            CubicSpline(q_values, np.log(beta_values), extrapolate=True)
        )

        # weightは後で正規化する前提でそのまま補間
        weight_splines.append(
            CubicSpline(q_values, w_values, extrapolate=True)
        )

    def global_cdf(x, q):
        x = np.asarray(x, dtype=float)

        eta = np.array([
            np.exp(spline(q)) for spline in eta_splines
        ])

        beta = np.array([
            np.exp(spline(q)) for spline in beta_splines
        ])

        weights = np.array([
            spline(q) for spline in weight_splines
        ], dtype=float)

        # weightが負になったり合計1からズレるのを防ぐ
        weights = np.clip(weights, 0, None)

        if weights.sum() == 0:
            weights = np.ones(n_components) / n_components
        else:
            weights = weights / weights.sum()

        F = np.zeros_like(x, dtype=float)

        for k in range(n_components):
            F += weights[k] * weibull_cdf(
                x,
                eta[k],
                beta[k]
            )

        return F

    return global_cdf

# # =========================
# # 可視化
# # =========================

# def plot_mixture_weibull_fit_compare(
#     df_fleetdata_long,
#     df_fit_results_dict,
#     n_components_list=(1, 2, 3),
#     width=1800,
#     height=600
# ):
#     fig = make_subplots(
#         rows=1,
#         cols=len(n_components_list),
#         subplot_titles=[
#             f"{n} component Weibull" for n in n_components_list
#         ],
#         shared_yaxes=False
#     )

#     for col_idx, n_components in enumerate(n_components_list, start=1):

#         df_fit_results = df_fit_results_dict.get(n_components)

#         # フィット結果が丸ごとない場合はスキップ
#         if df_fit_results is None or df_fit_results.empty:
#             fig.add_annotation(
#                 text=f"No fit result<br>{n_components} component",
#                 x=0.5,
#                 y=0.5,
#                 showarrow=False,
#                 row=1,
#                 col=col_idx
#             )
#             continue

#         for tp, df_group in df_fleetdata_long.groupby("ThroughputPercentile"):

#             fit_row = df_fit_results[
#                 df_fit_results["ThroughputPercentile"] == tp
#             ]

#             # そのThroughputPercentileのフィット結果がない場合はスキップ
#             if fit_row.empty:
#                 continue

#             params = fit_row.iloc[0].get("params", None)

#             # paramsがない、または欠損の場合はスキップ
#             if params is None:
#                 continue

#             try:
#                 params = np.asarray(params, dtype=float)
#             except Exception:
#                 continue

#             if params.size != 3 * n_components:
#                 continue

#             df_group = df_group.sort_values("value").copy()

#             x = df_group["value"].to_numpy(dtype=float)
#             p_obs = df_group["Percentile"].to_numpy(dtype=float)

#             mask = x > 0
#             x = x[mask]
#             p_obs = p_obs[mask]

#             if len(x) < 2:
#                 continue

#             x_grid = np.linspace(x.min(), x.max(), 300)

#             try:
#                 p_fit = mixture_weibull_cdf(
#                     x_grid,
#                     params,
#                     n_components=n_components
#                 )
#             except Exception:
#                 continue

#             # 実測点
#             fig.add_trace(
#                 go.Scatter(
#                     x=x,
#                     y=p_obs,
#                     mode="markers",
#                     name=f"Obs TP={tp}",
#                     legendgroup=f"TP={tp}",
#                     # showlegend=(col_idx == 1),
#                     showlegend=False,
#                     marker=dict(size=6)
#                 ),
#                 row=1,
#                 col=col_idx
#             )

#             # フィット線
#             fig.add_trace(
#                 go.Scatter(
#                     x=x_grid,
#                     y=p_fit,
#                     mode="lines",
#                     name=f"Fit TP={tp}",
#                     legendgroup=f"TP={tp}",
#                     showlegend=False
#                 ),
#                 row=1,
#                 col=col_idx
#             )

#         fig.update_xaxes(
#             title_text="DOD",
#             row=1,
#             col=col_idx
#         )

#         fig.update_yaxes(
#             title_text="Cumulative Probability",
#             row=1,
#             col=col_idx
#         )

#     fig.update_layout(
#         width=width,
#         height=height,
#         legend_title="Throughput Percentile"
#     )

#     return fig


# # =========================
# # QQプロット
# # =========================

# def mixture_weibull_ppf(p, params, n_components, x_min=1e-12, x_max=None):
#     """
#     混合ワイブルCDFの逆関数。
#     F(x) = p となる x を数値的に解く。
#     """

#     p = float(np.clip(p, 1e-10, 1 - 1e-10))

#     if x_max is None:
#         log_eta = params[0 : 2 * n_components : 2]
#         eta = np.exp(log_eta)
#         x_max = np.max(eta) * 100

#     def func(x):
#         return mixture_weibull_cdf(
#             np.array([x]),
#             params,
#             n_components
#         )[0] - p

#     # 上限が足りない場合に広げる
#     while func(x_max) < 0:
#         x_max *= 2

#     return brentq(func, x_min, x_max)


# def plot_mixture_weibull_qq_compare(
#     df_fleetdata_long,
#     df_fit_results_dict,
#     n_components_list=(1, 2, 3),
#     width=1800,
#     height=600
# ):
#     fig = make_subplots(
#         rows=1,
#         cols=len(n_components_list),
#         subplot_titles=[
#             f"{n} component Weibull QQ" for n in n_components_list
#         ],
#         shared_yaxes=False
#     )

#     for col_idx, n_components in enumerate(n_components_list, start=1):

#         df_fit_results = df_fit_results_dict.get(n_components)

#         if df_fit_results is None or df_fit_results.empty:
#             fig.add_annotation(
#                 text=f"No fit result<br>{n_components} component",
#                 x=0.5,
#                 y=0.5,
#                 showarrow=False,
#                 row=1,
#                 col=col_idx
#             )
#             continue

#         qq_x_all = []
#         qq_y_all = []

#         for tp, df_group in df_fleetdata_long.groupby("ThroughputPercentile"):

#             fit_row = df_fit_results[
#                 df_fit_results["ThroughputPercentile"] == tp
#             ]

#             if fit_row.empty:
#                 continue

#             params = fit_row.iloc[0].get("params", None)

#             if params is None:
#                 continue

#             try:
#                 params = np.asarray(params, dtype=float)
#             except Exception:
#                 continue

#             if params.size != 3 * n_components:
#                 continue

#             df_group = df_group.sort_values("value").copy()

#             x_obs = df_group["value"].to_numpy(dtype=float)
#             p_obs = df_group["Percentile"].to_numpy(dtype=float)

#             mask = (x_obs > 0) & (p_obs > 0) & (p_obs < 1)
#             x_obs = x_obs[mask]
#             p_obs = p_obs[mask]

#             if len(x_obs) < 2:
#                 continue

#             x_theory = []

#             for p in p_obs:
#                 try:
#                     q = mixture_weibull_ppf(
#                         p=p,
#                         params=params,
#                         n_components=n_components,
#                         x_max=max(x_obs.max() * 10, 1.0)
#                     )
#                     x_theory.append(q)
#                 except Exception:
#                     x_theory.append(np.nan)

#             x_theory = np.asarray(x_theory, dtype=float)

#             valid = np.isfinite(x_theory) & np.isfinite(x_obs)

#             if valid.sum() < 2:
#                 continue

#             x_theory = x_theory[valid]
#             x_obs_valid = x_obs[valid]

#             qq_x_all.extend(x_theory)
#             qq_y_all.extend(x_obs_valid)

#             fig.add_trace(
#                 go.Scatter(
#                     x=x_theory,
#                     y=x_obs_valid,
#                     mode="markers",
#                     name=f"TP={tp}",
#                     legendgroup=f"TP={tp}",
#                     # showlegend=(col_idx == 1)
#                     showlegend=False
#                 ),
#                 row=1,
#                 col=col_idx
#             )

#         # 45度線
#         if len(qq_x_all) > 0 and len(qq_y_all) > 0:
#             min_val = min(min(qq_x_all), min(qq_y_all))
#             max_val = max(max(qq_x_all), max(qq_y_all))

#             fig.add_trace(
#                 go.Scatter(
#                     x=[min_val, max_val],
#                     y=[min_val, max_val],
#                     mode="lines",
#                     name="45 degree line",
#                     showlegend=False,
#                     # showlegend=(col_idx == 1),
#                     line=dict(dash="dash")
#                 ),
#                 row=1,
#                 col=col_idx
#             )

#         fig.update_xaxes(
#             title_text="Theoretical quantile",
#             row=1,
#             col=col_idx
#         )

#         fig.update_yaxes(
#             title_text="Observed DOD",
#             row=1,
#             col=col_idx
#         )

#     fig.update_layout(
#         width=width,
#         height=height,
#         # title="QQ plot",
#         legend_title="Throughput Percentile"
#     )

#     return fig


# =========================
# RSME
# =========================

def safe_mean_rmse(df):
    if df is None:
        return np.nan
    
    if not isinstance(df, pd.DataFrame):
        return np.nan
    
    if df.empty:
        return np.nan
    
    if "rmse" not in df.columns:
        return np.nan
    
    return df["rmse"].mean()
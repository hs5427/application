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
# 混合ワイブルCDF
# =========================

def mixture_weibull_cdf(x, params, n_components):

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


# =========================
# 残差計算
# =========================

def residuals(params, x, p_obs, n_components):
    p_pred = mixture_weibull_cdf(x, params, n_components)
    return p_obs - p_pred


# =========================
# 初期値作成
# =========================

def make_initial_params(x, n_components):

    if n_components == 1:
        quantiles = [50]
    elif n_components == 2:
        quantiles = [30, 80]
    elif n_components == 3:
        quantiles = [10, 50, 90]
    else:
        quantiles = np.linspace(10, 90, n_components)

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
# Fitting
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
    }

    for k in range(n_components):
        output[f"eta{k+1}"] = eta[k]
        output[f"beta{k+1}"] = beta[k]
        output[f"w{k+1}"] = weights[k]

    return output

# =========================
# ThroughputPercentileごとにFitting
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
    q_col="TTP_Percentile",
    param_fit_method="spline"
):

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

        eta_splines.append(
            make_param_function(
                q_values,
                np.log(eta_values),
                method=param_fit_method
            )
        )

        beta_splines.append(
            make_param_function(
                q_values,
                np.log(beta_values),
                method=param_fit_method
            )
        )

        weight_splines.append(
            make_param_function(
                q_values,
                w_values,
                method=param_fit_method
            )
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


def make_param_function(q_values, y_values, method="spline"):
    q_values = np.asarray(q_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if method == "spline":
        return CubicSpline(
            q_values,
            y_values,
            extrapolate=True
        )

    elif method == "poly3":
        degree = min(3, len(q_values) - 1)
        coef = np.polyfit(
            q_values,
            y_values,
            deg=degree
        )

        return np.poly1d(coef)

    else:
        raise ValueError(
            "param_fit_method must be 'spline' or 'poly3'."
        )
    
    
def make_weibull_param_coefficients(
    df_fit,
    n_components=2,
    q_col="TTP_Percentile",
    method="poly3"
):

    if df_fit is None or df_fit.empty:
        return pd.DataFrame()

    df_fit = df_fit.sort_values(q_col).copy()
    q = df_fit[q_col].to_numpy(dtype=float)

    rows = []

    for k in range(1, n_components + 1):

        param_cols = [
            f"eta{k}",
            f"beta{k}",
            f"w{k}"
        ]

        for param_col in param_cols:

            if param_col not in df_fit.columns:
                continue

            y = df_fit[param_col].to_numpy(dtype=float)

            if method == "poly3":

                degree = min(3, len(q) - 1)
                coef = np.polyfit(q, y, deg=degree)

                # np.polyfitは高次 → 低次の順
                # degree=3なら [a3, a2, a1, a0]
                coef_full = np.full(4, np.nan)
                coef_full[-len(coef):] = coef

                rows.append({
                    "method": "poly3",
                    "parameter": param_col,
                    "interval_start": np.nan,
                    "interval_end": np.nan,
                    "formula": f"{param_col}(q) = a3*q^3 + a2*q^2 + a1*q + a0",
                    "a3": coef_full[0],
                    "a2": coef_full[1],
                    "a1": coef_full[2],
                    "a0": coef_full[3],
                })

            elif method == "spline":

                spline = CubicSpline(
                    q,
                    y,
                    extrapolate=True
                )

                # spline.c shape = (4, n_intervals)
                # 各区間で:
                # y = c0*(q-q_i)^3 + c1*(q-q_i)^2 + c2*(q-q_i) + c3
                c = spline.c

                for i in range(len(q) - 1):
                    rows.append({
                        "method": "spline",
                        "parameter": param_col,
                        "interval_start": q[i],
                        "interval_end": q[i + 1],
                        "formula": (
                            f"{param_col}(q) = "
                            f"a3*(q-{q[i]})^3 + "
                            f"a2*(q-{q[i]})^2 + "
                            f"a1*(q-{q[i]}) + a0"
                        ),
                        "a3": c[0, i],
                        "a2": c[1, i],
                        "a1": c[2, i],
                        "a0": c[3, i],
                    })

            else:
                raise ValueError(
                    "method must be 'poly3' or 'spline'."
                )

    return pd.DataFrame(rows)


def plot_weibull_parameter_trends(
    df_fit,
    n_components=2,
    q_col="TTP_Percentile",
    param_fit_method="spline",
    n_grid=300
):
    """
    TTP_Percentileごとの eta, beta, w の実測点と
    TTP方向にフィットした曲線を可視化する。
    """

    if df_fit is None or df_fit.empty:
        return None

    df_fit = df_fit.sort_values(q_col).copy()

    q_values = df_fit[q_col].to_numpy(dtype=float)
    q_grid = np.linspace(q_values.min(), q_values.max(), n_grid)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "eta parameter",
            "beta parameter",
            "weight parameter"
        ]
    )

    param_groups = [
        ("eta", 1),
        ("beta", 2),
        ("w", 3)
    ]

    for param_prefix, col_idx in param_groups:

        for k in range(1, n_components + 1):

            param_col = f"{param_prefix}{k}"

            if param_col not in df_fit.columns:
                continue

            y_values = df_fit[param_col].to_numpy(dtype=float)

            # =========================
            # Scatter: fitted parameters
            # =========================
            fig.add_trace(
                go.Scatter(
                    x=q_values,
                    y=y_values,
                    mode="markers",
                    name=f"{param_col} points",
                    legendgroup=param_col,
                    showlegend=True
                ),
                row=1,
                col=col_idx
            )

            # =========================
            # Fit curve
            # =========================
            fit_func = make_param_function(
                q_values,
                y_values,
                method=param_fit_method
            )

            y_fit = fit_func(q_grid)

            if param_prefix == "w":
                y_fit = np.clip(y_fit, 0, None)

            fig.add_trace(
                go.Scatter(
                    x=q_grid,
                    y=y_fit,
                    mode="lines",
                    name=f"{param_col} fit",
                    legendgroup=param_col,
                    showlegend=True
                ),
                row=1,
                col=col_idx
            )

    fig.update_xaxes(title_text="TTP Percentile", row=1, col=1)
    fig.update_xaxes(title_text="TTP Percentile", row=1, col=2)
    fig.update_xaxes(title_text="TTP Percentile", row=1, col=3)

    fig.update_yaxes(title_text="eta", row=1, col=1)
    fig.update_yaxes(title_text="beta", row=1, col=2)
    fig.update_yaxes(title_text="weight", row=1, col=3)

    fig.update_layout(
        height=550,
        width=1500,
        title="Weibull parameter trends with fitted curves",
        legend_title="Parameter"
    )

    return fig


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


# =========================
# 非従属パラメータのFitting
# =========================

def fit_independent_mixture_weibull(
    df_long,
    n_components=2,
    p_col="value_Percentile",
    value_col="value"
):
    """
    TTPに依存しない単独分布を混合ワイブルでフィットする。
    """

    df_group = df_long[[p_col, value_col]].copy()

    df_group = df_group.rename(columns={
        p_col: "value_Percentile",
        value_col: "value"
    })

    fit = fit_mixture_weibull(
        df_group,
        n_components=n_components
    )

    if fit is None:
        return pd.DataFrame()

    return pd.DataFrame([fit])


def build_independent_mixture_weibull_cdf(
    df_fit,
    n_components=2
):
    """
    TTPに依存しない混合ワイブルCDFを作成する。
    返り値は global_cdf(x, q=None)
    """

    if df_fit is None or df_fit.empty:
        raise ValueError("フィット結果がありません。")

    row = df_fit.iloc[0]

    eta = np.array([
        row[f"eta{k}"]
        for k in range(1, n_components + 1)
    ], dtype=float)

    beta = np.array([
        row[f"beta{k}"]
        for k in range(1, n_components + 1)
    ], dtype=float)

    weights = np.array([
        row[f"w{k}"]
        for k in range(1, n_components + 1)
    ], dtype=float)

    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()

    def global_cdf(x, q=None):
        x = np.asarray(x, dtype=float)

        F = np.zeros_like(x, dtype=float)

        for k in range(n_components):
            F += weights[k] * weibull_cdf(
                x,
                eta[k],
                beta[k]
            )

        return F

    return global_cdf
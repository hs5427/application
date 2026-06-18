import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from scipy.special import erf
from scipy.interpolate import CubicSpline


# =========================
# Gaussian CDF
# =========================

def gaussian_cdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    z = (x - mu) / (sigma * np.sqrt(2))
    return 0.5 * (1.0 + erf(z))


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


# =========================
# K成分混合正規CDF
# =========================

def mixture_gaussian_cdf(x, params, n_components):
    """
    params:
        [mu1, log_sigma1,
         mu2, log_sigma2,
         ...,
         z_weight1, z_weight2, ...]
    """

    x = np.asarray(x, dtype=float)

    mu = params[0 : 2 * n_components : 2]
    log_sigma = params[1 : 2 * n_components : 2]
    z_weight = params[2 * n_components : 3 * n_components]

    sigma = np.exp(log_sigma)
    weights = softmax(z_weight)

    F = np.zeros_like(x, dtype=float)

    for k in range(n_components):
        F += weights[k] * gaussian_cdf(
            x,
            mu[k],
            sigma[k]
        )

    return F


def residuals(params, x, p_obs, n_components):
    p_pred = mixture_gaussian_cdf(
        x,
        params,
        n_components
    )

    return p_obs - p_pred


# =========================
# 初期値作成
# =========================

def make_initial_params(x, n_components):
    x = np.asarray(x, dtype=float)

    if n_components == 1:
        quantiles = [50]
    elif n_components == 2:
        quantiles = [30, 80]
    elif n_components == 3:
        quantiles = [10, 50, 90]
    else:
        quantiles = np.linspace(10, 90, n_components)

    mu_init = np.percentile(x, quantiles)

    x_std = np.nanstd(x)
    if not np.isfinite(x_std) or x_std <= 0:
        x_std = max(np.nanmax(x) - np.nanmin(x), 1.0)

    sigma_init = np.full(
        n_components,
        x_std / max(n_components, 1)
    )

    sigma_init = np.clip(sigma_init, 1e-12, None)

    log_sigma_init = np.log(sigma_init)

    z_weight_init = np.zeros(n_components)

    params = []

    for k in range(n_components):
        params.append(mu_init[k])
        params.append(log_sigma_init[k])

    params.extend(z_weight_init)

    return np.array(params, dtype=float)


# =========================
# bounds作成
# =========================

def make_bounds(x, n_components):
    x = np.asarray(x, dtype=float)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_range = x_max - x_min

    if not np.isfinite(x_range) or x_range <= 0:
        x_range = max(abs(x_max), 1.0)

    lower = []
    upper = []

    for _ in range(n_components):
        # mu
        lower.append(x_min - 5 * x_range)
        upper.append(x_max + 5 * x_range)

        # sigma
        lower.append(np.log(max(x_range * 1e-6, 1e-12)))
        upper.append(np.log(max(x_range * 10, 1.0)))

    # mixture weight用のz
    for _ in range(n_components):
        lower.append(-20)
        upper.append(20)

    return np.asarray(lower), np.asarray(upper)


# =========================
# 1グループをK成分混合正規でフィット
# =========================

def fit_mixture_gaussian(df_group, n_components=2):
    df_group = df_group.sort_values("value").copy()

    x = df_group["value"].to_numpy(dtype=float)
    p_obs = df_group["value_Percentile"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(p_obs)
    x = x[mask]
    p_obs = p_obs[mask]

    p_obs = np.clip(p_obs, 1e-6, 1 - 1e-6)

    # mu, sigma, weight で合計 3K パラメータ
    if len(x) < 3 * n_components:
        return None

    init_params = make_initial_params(
        x,
        n_components
    )

    bounds = make_bounds(
        x,
        n_components
    )

    result = least_squares(
        residuals,
        x0=init_params,
        args=(x, p_obs, n_components),
        bounds=bounds,
        max_nfev=20000
    )

    params_hat = result.x

    mu = params_hat[0 : 2 * n_components : 2]
    log_sigma = params_hat[1 : 2 * n_components : 2]
    z_weight = params_hat[2 * n_components : 3 * n_components]

    sigma = np.exp(log_sigma)
    weights = softmax(z_weight)

    p_pred = mixture_gaussian_cdf(
        x,
        params_hat,
        n_components
    )

    rmse = np.sqrt(
        np.mean((p_obs - p_pred) ** 2)
    )

    sse = np.sum((p_obs - p_pred) ** 2)
    sst = np.sum((p_obs - np.mean(p_obs)) ** 2)

    if sst == 0:
        r2 = np.nan
    else:
        r2 = 1 - sse / sst

    output = {
        "n_components": n_components,
        "rmse": rmse,
        "r2": r2,
        "success": result.success,
        "message": result.message
    }

    for k in range(n_components):
        output[f"mu{k+1}"] = mu[k]
        output[f"sigma{k+1}"] = sigma[k]
        output[f"w{k+1}"] = weights[k]

    return output


# =========================
# TTP_Percentileごとにフィット
# =========================

def fit_gaussian_by_throughput_percentile(
    df_long,
    n_components=2
):
    fit_results = []

    for tp, df_group in df_long.groupby("TTP_Percentile"):
        try:
            fit = fit_mixture_gaussian(
                df_group,
                n_components=n_components
            )

            if fit is None:
                print(
                    f"Skipped TTP_Percentile={tp}: insufficient data"
                )
                continue

            fit["TTP_Percentile"] = tp
            fit_results.append(fit)

        except Exception as e:
            print(
                f"Gaussian fitting failed at TTP_Percentile={tp}: {e}"
            )
            continue

    return pd.DataFrame(fit_results)


# =========================
# Independent fit
# =========================

def fit_independent_mixture_gaussian(
    df_long,
    n_components=2,
    p_col="value_Percentile",
    value_col="value"
):
    df_group = df_long[[p_col, value_col]].copy()

    df_group = df_group.rename(columns={
        p_col: "value_Percentile",
        value_col: "value"
    })

    fit = fit_mixture_gaussian(
        df_group,
        n_components=n_components
    )

    if fit is None:
        return pd.DataFrame()

    return pd.DataFrame([fit])


# =========================
# parameter interpolation
# =========================

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


# =========================
# Global CDF: TTP dependent
# =========================

def build_global_mixture_gaussian_cdf(
    df_fit_results_dict,
    n_components=2,
    q_col="TTP_Percentile",
    param_fit_method="spline"
):
    df_fit = df_fit_results_dict.get(n_components)

    if df_fit is None or df_fit.empty:
        raise ValueError(
            f"{n_components}成分の混合正規フィット結果がありません。"
        )

    df_fit = df_fit.sort_values(q_col).copy()

    q_values = df_fit[q_col].to_numpy(dtype=float)

    mu_funcs = []
    sigma_funcs = []
    weight_funcs = []

    for k in range(1, n_components + 1):
        mu_col = f"mu{k}"
        sigma_col = f"sigma{k}"
        w_col = f"w{k}"

        if mu_col not in df_fit.columns:
            raise ValueError(f"{mu_col} が df_fit にありません。")
        if sigma_col not in df_fit.columns:
            raise ValueError(f"{sigma_col} が df_fit にありません。")
        if w_col not in df_fit.columns:
            raise ValueError(f"{w_col} が df_fit にありません。")

        mu_values = df_fit[mu_col].to_numpy(dtype=float)
        sigma_values = df_fit[sigma_col].to_numpy(dtype=float)
        w_values = df_fit[w_col].to_numpy(dtype=float)

        mu_funcs.append(
            make_param_function(
                q_values,
                mu_values,
                method=param_fit_method
            )
        )

        sigma_funcs.append(
            make_param_function(
                q_values,
                np.log(sigma_values),
                method=param_fit_method
            )
        )

        weight_funcs.append(
            make_param_function(
                q_values,
                w_values,
                method=param_fit_method
            )
        )

    def global_cdf(x, q):
        x = np.asarray(x, dtype=float)

        mu = np.array([
            func(q) for func in mu_funcs
        ], dtype=float)

        sigma = np.array([
            np.exp(func(q)) for func in sigma_funcs
        ], dtype=float)

        weights = np.array([
            func(q) for func in weight_funcs
        ], dtype=float)

        weights = np.clip(weights, 0, None)

        if weights.sum() == 0:
            weights = np.ones(n_components) / n_components
        else:
            weights = weights / weights.sum()

        F = np.zeros_like(x, dtype=float)

        for k in range(n_components):
            F += weights[k] * gaussian_cdf(
                x,
                mu[k],
                sigma[k]
            )

        return F

    return global_cdf


# =========================
# Global CDF: Independent
# =========================

def build_independent_mixture_gaussian_cdf(
    df_fit,
    n_components=2
):
    if df_fit is None or df_fit.empty:
        raise ValueError("混合正規のフィット結果がありません。")

    row = df_fit.iloc[0]

    mu = np.array([
        row[f"mu{k}"]
        for k in range(1, n_components + 1)
    ], dtype=float)

    sigma = np.array([
        row[f"sigma{k}"]
        for k in range(1, n_components + 1)
    ], dtype=float)

    weights = np.array([
        row[f"w{k}"]
        for k in range(1, n_components + 1)
    ], dtype=float)

    weights = np.clip(weights, 0, None)

    if weights.sum() == 0:
        weights = np.ones(n_components) / n_components
    else:
        weights = weights / weights.sum()

    def global_cdf(x, q=None):
        x = np.asarray(x, dtype=float)

        F = np.zeros_like(x, dtype=float)

        for k in range(n_components):
            F += weights[k] * gaussian_cdf(
                x,
                mu[k],
                sigma[k]
            )

        return F

    return global_cdf


import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_gaussian_param_coefficients(
    df_fit,
    n_components=2,
    q_col="TTP_Percentile",
    method="spline"
):
    """
    混合正規分布パラメータ mu, sigma, weight の
    TTP方向フィット情報をテーブル化する。
    """

    if df_fit is None or df_fit.empty:
        return pd.DataFrame()

    if q_col not in df_fit.columns:
        return pd.DataFrame()

    df_fit = df_fit.sort_values(q_col).copy()

    q = df_fit[q_col].to_numpy(dtype=float)

    rows = []

    for k in range(1, n_components + 1):

        param_cols = {
            f"mu{k}": f"mu{k}",
            f"sigma{k}": f"sigma{k}",
            f"w{k}": f"w{k}"
        }

        for param_name, col in param_cols.items():

            if col not in df_fit.columns:
                continue

            y = df_fit[col].to_numpy(dtype=float)

            if method == "poly3":
                degree = min(3, len(q) - 1)
                coef = np.polyfit(q, y, deg=degree)

                row = {
                    "component": k,
                    "parameter": param_name,
                    "method": "poly3",
                    "degree": degree
                }

                for i, c in enumerate(coef):
                    row[f"coef_{i}"] = c

                rows.append(row)

            elif method == "spline":
                rows.append({
                    "component": k,
                    "parameter": param_name,
                    "method": "spline",
                    "note": "CubicSpline interpolation. Coefficients are piecewise and not summarized as one global polynomial."
                })

            else:
                raise ValueError("method must be 'spline' or 'poly3'.")

    return pd.DataFrame(rows)


def plot_gaussian_parameter_trends(
    df_fit,
    n_components=2,
    q_col="TTP_Percentile",
    param_fit_method="spline"
):
    """
    混合正規分布パラメータ mu, sigma, weight の
    TTP方向トレンドを描画する。
    """

    if df_fit is None or df_fit.empty:
        return None

    if q_col not in df_fit.columns:
        return None

    df_fit = df_fit.sort_values(q_col).copy()

    q = df_fit[q_col].to_numpy(dtype=float)

    q_grid = np.linspace(
        float(np.min(q)),
        float(np.max(q)),
        300
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "mu trend",
            "sigma trend",
            "weight trend"
        ],
        vertical_spacing=0.12
    )

    param_specs = [
        ("mu", 1),
        ("sigma", 2),
        ("w", 3)
    ]

    for param_prefix, row in param_specs:

        for k in range(1, n_components + 1):

            col = f"{param_prefix}{k}"

            if col not in df_fit.columns:
                continue

            y = df_fit[col].to_numpy(dtype=float)

            fig.add_trace(
                go.Scatter(
                    x=q,
                    y=y,
                    mode="markers",
                    name=f"{col} data"
                ),
                row=row,
                col=1
            )

            if len(q) >= 2:

                if param_fit_method == "poly3":
                    degree = min(3, len(q) - 1)
                    coef = np.polyfit(q, y, deg=degree)
                    y_fit = np.polyval(coef, q_grid)

                elif param_fit_method == "spline":
                    func = CubicSpline(
                        q,
                        y,
                        extrapolate=True
                    )
                    y_fit = func(q_grid)

                else:
                    raise ValueError(
                        "param_fit_method must be 'spline' or 'poly3'."
                    )

                fig.add_trace(
                    go.Scatter(
                        x=q_grid,
                        y=y_fit,
                        mode="lines",
                        name=f"{col} {param_fit_method}"
                    ),
                    row=row,
                    col=1
                )

        fig.update_yaxes(
            title_text=param_prefix,
            row=row,
            col=1
        )

    fig.update_xaxes(
        title_text="TTP_Percentile",
        row=3,
        col=1
    )

    fig.update_layout(
        height=900,
        title="Gaussian mixture parameter trends",
        showlegend=True
    )

    return fig
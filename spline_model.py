import numpy as np
import pandas as pd

from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from scipy.optimize import brentq


# =========================
# Independent spline model
# =========================

def fit_independent_spline_model(
    df_long,
    p_col="value_Percentile",
    value_col="value"
):
    df = df_long[[p_col, value_col]].dropna().copy()

    df[p_col] = df[p_col].astype(float)
    df[value_col] = df[value_col].astype(float)

    df = df.sort_values(p_col)

    p = df[p_col].to_numpy(dtype=float)
    x = df[value_col].to_numpy(dtype=float)

    p_unique, idx = np.unique(p, return_index=True)
    x_unique = x[idx]

    if len(p_unique) < 2:
        raise ValueError("Independent spline model requires at least 2 percentile points.")

    ppf_func = PchipInterpolator(
        p_unique,
        x_unique,
        extrapolate=True
    )

    def global_ppf(p_value, q=None):
        p_value = np.asarray(p_value, dtype=float)
        p_value = np.clip(p_value, p_unique.min(), p_unique.max())
        return ppf_func(p_value)

    def global_cdf(x_value, q=None):
        x_value = np.asarray(x_value, dtype=float)

        x_min = float(np.nanmin(x_unique))
        x_max = float(np.nanmax(x_unique))

        def solve_one(x0):
            def f(p0):
                return float(global_ppf(p0)) - float(x0)

            if x0 <= x_min:
                return float(p_unique.min())
            if x0 >= x_max:
                return float(p_unique.max())

            return brentq(
                f,
                float(p_unique.min()),
                float(p_unique.max())
            )

        if x_value.ndim == 0:
            return solve_one(float(x_value))

        return np.array([
            solve_one(v)
            for v in x_value
        ])

    df_fit = pd.DataFrame({
        "model": ["spline_1d"],
        "n_points": [len(p_unique)],
        "p_min": [p_unique.min()],
        "p_max": [p_unique.max()],
        "value_min": [x_unique.min()],
        "value_max": [x_unique.max()]
    })

    return {
        "df_fit": df_fit,
        "global_cdf": global_cdf,
        "global_ppf": global_ppf
    }


# =========================
# TTP dependent spline model
# =========================

def fit_dependent_spline_model(
    df_long,
    q_col="TTP_Percentile",
    p_col="value_Percentile",
    value_col="value",
    kx=3,
    ky=3
):
    df = df_long[[q_col, p_col, value_col]].dropna().copy()

    df[q_col] = df[q_col].astype(float)
    df[p_col] = df[p_col].astype(float)
    df[value_col] = df[value_col].astype(float)

    q_values = np.sort(df[q_col].unique())
    p_values = np.sort(df[p_col].unique())

    if len(q_values) < 2:
        raise ValueError("2D spline requires at least 2 TTP_Percentile values.")
    if len(p_values) < 2:
        raise ValueError("2D spline requires at least 2 value_Percentile values.")

    value_grid = (
        df
        .pivot_table(
            index=q_col,
            columns=p_col,
            values=value_col,
            aggfunc="mean"
        )
        .reindex(index=q_values, columns=p_values)
    )

    if value_grid.isna().any().any():
        raise ValueError(
            "2D spline requires a complete grid of "
            "TTP_Percentile x value_Percentile."
        )

    z = value_grid.to_numpy(dtype=float)

    kx_eff = min(kx, len(q_values) - 1)
    ky_eff = min(ky, len(p_values) - 1)

    spline_2d = RectBivariateSpline(
        q_values,
        p_values,
        z,
        kx=kx_eff,
        ky=ky_eff
    )

    def global_ppf(p_value, q):
        p_value = np.asarray(p_value, dtype=float)
        p_value = np.clip(p_value, p_values.min(), p_values.max())

        q = float(np.clip(q, q_values.min(), q_values.max()))

        return spline_2d(
            q,
            p_value,
            grid=False
        )

    def global_cdf(x_value, q):
        x_value = np.asarray(x_value, dtype=float)
        q = float(np.clip(q, q_values.min(), q_values.max()))

        x_min = float(global_ppf(p_values.min(), q))
        x_max = float(global_ppf(p_values.max(), q))

        def solve_one(x0):
            def f(p0):
                return float(global_ppf(p0, q)) - float(x0)

            if x0 <= x_min:
                return float(p_values.min())
            if x0 >= x_max:
                return float(p_values.max())

            return brentq(
                f,
                float(p_values.min()),
                float(p_values.max())
            )

        if x_value.ndim == 0:
            return solve_one(float(x_value))

        return np.array([
            solve_one(v)
            for v in x_value
        ])

    df_fit = pd.DataFrame({
        "model": ["spline_2d"],
        "n_q_points": [len(q_values)],
        "n_p_points": [len(p_values)],
        "q_min": [q_values.min()],
        "q_max": [q_values.max()],
        "p_min": [p_values.min()],
        "p_max": [p_values.max()],
        "kx": [kx_eff],
        "ky": [ky_eff]
    })

    return {
        "df_fit": df_fit,
        "global_cdf": global_cdf,
        "global_ppf": global_ppf
    }


# =========================
# Sampler for spline model
# =========================

def build_spline_sampler(
    global_ppf,
    analysis_mode="Independent",
    df_throughput_quantile=None,
    q_col="TTP_Percentile",
    throughput_col="TTP"
):
    if analysis_mode == "TTP dependent":

        if df_throughput_quantile is None:
            raise ValueError("TTP dependent spline sampler requires df_throughput_quantile.")

        df_q = df_throughput_quantile[[q_col, throughput_col]].dropna().copy()
        df_q = df_q.sort_values(q_col)

        q_values = df_q[q_col].to_numpy(dtype=float)
        t_values = df_q[throughput_col].to_numpy(dtype=float)

        if q_values.max() > 1:
            q_prob = q_values / 100
        else:
            q_prob = q_values.copy()

        ttp_ppf = PchipInterpolator(
            q_prob,
            t_values,
            extrapolate=True
        )

    def sample(
        n_samples=10000,
        random_state=42
    ):
        rng = np.random.default_rng(random_state)

        u = rng.uniform(
            1e-10,
            1 - 1e-10,
            size=n_samples
        )

        if analysis_mode == "TTP dependent":

            q_random_prob = rng.uniform(
                q_prob.min(),
                q_prob.max(),
                size=n_samples
            )

            ttp_samples = ttp_ppf(q_random_prob)

            if q_values.max() > 1:
                q_for_model = q_random_prob * 100
            else:
                q_for_model = q_random_prob

            values = np.array([
                float(global_ppf(p_value, q_value))
                for p_value, q_value in zip(u, q_for_model)
            ])

            return pd.DataFrame({
                "TTP_Percentile": q_for_model,
                "TTP": ttp_samples,
                "value": values
            })

        else:
            values = np.array([
                float(global_ppf(p_value))
                for p_value in u
            ])

            return pd.DataFrame({
                "value": values
            })

    return sample
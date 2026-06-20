import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def global_cdf_ppf(
    global_cdf,
    p,
    q=None,
    model_family="weibull",
    x_min_init=None,
    x_max_init=None,
    max_expand=100,
    expand_rate=2.0
):

    p = float(np.clip(p, 1e-10, 1 - 1e-10))

    # =========================
    # Initial bracket by model
    # =========================

    if x_min_init is not None:
        x_min = float(x_min_init)
    else:
        if model_family == "weibull":
            x_min = 1e-12
        elif model_family == "gaussian":
            x_min = -1.0
        else:
            x_min = 1e-12

    if x_max_init is not None:
        x_max = float(x_max_init)
    else:
        x_max = 1.0

    def f(x):
        val = float(global_cdf(x, q)) - p

        if not np.isfinite(val):
            return np.nan

        return val

    f_min = f(x_min)
    f_max = f(x_max)

    # =========================
    # Weibull: never expand to negative side
    # =========================

    if model_family == "weibull":

        if not np.isfinite(f_min):
            x_min = 1e-12
            f_min = f(x_min)

        expand_count = 0

        while (
            (not np.isfinite(f_max) or f_max < 0)
            and expand_count < max_expand
        ):
            x_max = x_max * expand_rate

            if x_max <= x_min:
                x_max = x_min + 1.0

            f_max = f(x_max)
            expand_count += 1

        if not np.isfinite(f_min) or not np.isfinite(f_max):
            raise ValueError(
                f"Weibull inverse CDF failed due to non-finite bracket. "
                f"p={p}, q={q}, x_min={x_min}, f_min={f_min}, "
                f"x_max={x_max}, f_max={f_max}"
            )

        if f_min * f_max > 0:
            raise ValueError(
                f"Weibull inverse CDF failed to bracket root. "
                f"p={p}, q={q}, x_min={x_min}, f_min={f_min}, "
                f"x_max={x_max}, f_max={f_max}"
            )

        return brentq(
            f,
            x_min,
            x_max,
            maxiter=200
        )

    # =========================
    # Gaussian: expand both sides
    # =========================

    elif model_family == "gaussian":

        expand_count = 0

        while (
            (not np.isfinite(f_min) or f_min > 0)
            and expand_count < max_expand
        ):
            width = x_max - x_min

            if width <= 0:
                width = 1.0

            x_min = x_min - expand_rate * width
            f_min = f(x_min)

            expand_count += 1

        expand_count = 0

        while (
            (not np.isfinite(f_max) or f_max < 0)
            and expand_count < max_expand
        ):
            width = x_max - x_min

            if width <= 0:
                width = 1.0

            x_max = x_max + expand_rate * width
            f_max = f(x_max)

            expand_count += 1

        if not np.isfinite(f_min) or not np.isfinite(f_max):
            raise ValueError(
                f"Gaussian inverse CDF failed due to non-finite bracket. "
                f"p={p}, q={q}, x_min={x_min}, f_min={f_min}, "
                f"x_max={x_max}, f_max={f_max}"
            )

        if f_min * f_max > 0:
            raise ValueError(
                f"Gaussian inverse CDF failed to bracket root. "
                f"p={p}, q={q}, x_min={x_min}, f_min={f_min}, "
                f"x_max={x_max}, f_max={f_max}"
            )

        return brentq(
            f,
            x_min,
            x_max,
            maxiter=200
        )

    else:
        raise ValueError(
            "model_family must be 'weibull' or 'gaussian'."
        )


def build_bivariate_sampler(
    df_throughput_quantile,
    global_cdf,
    q_col="TTP_Percentile",
    throughput_col="TTP",
    model_family="weibull"
):
    df_q = df_throughput_quantile[[q_col, throughput_col]].dropna().copy()
    df_q = df_q.sort_values(q_col)

    q_values = df_q[q_col].to_numpy(dtype=float)
    t_values = df_q[throughput_col].to_numpy(dtype=float)

    if q_values.max() > 1:
        q_prob = q_values / 100
    else:
        q_prob = q_values.copy()

    # TTPパーセンタイル -> TTPを予測する関数
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

        # TTPをサンプリング
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

        samples = []

        # パラメータをサンプリング
        for q in q_for_model:

            u = rng.uniform(1e-10, 1 - 1e-10)

            feature_value = global_cdf_ppf(
                global_cdf=global_cdf,
                p=u,
                q=q,
                model_family=model_family
            )

            samples.append(feature_value)

        df_sample = pd.DataFrame({
            "TTP_Percentile": q_for_model,
            "TTP": ttp_samples,
            "value": samples
        })

        return df_sample

    return sample


def build_independent_sampler(
    global_cdf,
    model_family="weibull"
):

    def sample(
        n_samples=10000,
        random_state=42
    ):
        rng = np.random.default_rng(random_state)

        u_values = rng.uniform(
            1e-10,
            1 - 1e-10,
            size=n_samples
        )

        values = []

        for u in u_values:
            x = global_cdf_ppf(
                global_cdf=global_cdf,
                p=u,
                q=None,
                model_family=model_family
            )

            values.append(x)

        df_sample = pd.DataFrame({
            "value": values
        })

        return df_sample

    return sample
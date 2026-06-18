import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def global_cdf_ppf(
    global_cdf,
    p,
    q,
    x_min=1e-12,
    x_max_init=1.0
):
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    upper = x_max_init

    def func(x):
        return float(global_cdf(x, q)) - p

    while func(upper) < 0:
        upper *= 2

        if upper > 1e12:
            raise RuntimeError("探索上限が大きくなりすぎました。")

    return brentq(func, x_min, upper)


def build_bivariate_sampler(
    df_throughput_quantile,
    global_cdf,
    q_col="TTP_Percentile",
    throughput_col="TTP"
):
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

        x_max_init = 1.0

        for q in q_for_model:

            u = rng.uniform(1e-10, 1 - 1e-10)

            feature_value = global_cdf_ppf(
                global_cdf=global_cdf,
                p=u,
                q=q,
                x_max_init=x_max_init
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
    global_cdf
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
                x_max_init=1.0
            )

            values.append(x)

        df_sample = pd.DataFrame({
            "value": values
        })

        return df_sample

    return sample
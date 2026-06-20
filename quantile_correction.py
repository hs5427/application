import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def apply_quantile_correction(
    df_sample,
    df_reference,
    global_cdf,
    q_col="TTP_Percentile",
    p_col="value_Percentile",
    value_col="value"
):

    df_sample = df_sample.copy()
    df_ref = df_reference[[q_col, p_col, value_col]].dropna().copy()

    q_ref_values = np.sort(df_ref[q_col].unique())
    p_ref_values = np.sort(df_ref[p_col].unique())

    # pごとに TTP方向の value 補間関数を作る
    value_funcs_by_p = {}

    for p in p_ref_values:
        df_p = df_ref[df_ref[p_col] == p].sort_values(q_col)

        if len(df_p) < 2:
            continue

        value_funcs_by_p[p] = PchipInterpolator(
            df_p[q_col].to_numpy(dtype=float),
            df_p[value_col].to_numpy(dtype=float),
            extrapolate=True
        )

    corrected_values = []

    for _, row in df_sample.iterrows():
        q = float(row[q_col])
        x = float(row[value_col])

        # 生成値がモデル上で何パーセンタイルか
        u = float(global_cdf(x, q))
        u = np.clip(u, p_ref_values.min(), p_ref_values.max())

        # そのqにおける元データ側の分位点カーブを作る
        p_list = []
        x_ref_list = []

        for p, func in value_funcs_by_p.items():
            p_list.append(p)
            x_ref_list.append(float(func(q)))

        p_arr = np.asarray(p_list, dtype=float)
        x_ref_arr = np.asarray(x_ref_list, dtype=float)

        order = np.argsort(p_arr)
        p_arr = p_arr[order]
        x_ref_arr = x_ref_arr[order]

        ppf_ref = PchipInterpolator(
            p_arr,
            x_ref_arr,
            extrapolate=True
        )

        corrected_x = float(ppf_ref(u))
        corrected_values.append(corrected_x)

    df_sample[f"{value_col}_raw"] = df_sample[value_col]
    df_sample[value_col] = corrected_values

    return df_sample


def apply_independent_quantile_correction(
    df_sample,
    df_reference,
    global_cdf,
    p_col="value_Percentile",
    value_col="value"
):


    df_sample = df_sample.copy()
    df_ref = df_reference[[p_col, value_col]].dropna().copy()

    df_ref = df_ref.sort_values(p_col)

    p_ref = df_ref[p_col].to_numpy(dtype=float)
    x_ref = df_ref[value_col].to_numpy(dtype=float)

    order = np.argsort(p_ref)
    p_ref = p_ref[order]
    x_ref = x_ref[order]

    p_unique, unique_idx = np.unique(p_ref, return_index=True)
    x_unique = x_ref[unique_idx]

    if len(p_unique) < 2:
        raise ValueError("Independent quantile correction requires at least 2 percentile points.")

    ppf_ref = PchipInterpolator(
        p_unique,
        x_unique,
        extrapolate=True
    )

    corrected_values = []

    for x in df_sample[value_col].to_numpy(dtype=float):
        u = float(global_cdf(x, q=None))
        u = np.clip(u, p_unique.min(), p_unique.max())

        corrected_x = float(ppf_ref(u))
        corrected_values.append(corrected_x)

    df_sample[f"{value_col}_raw"] = df_sample[value_col]
    df_sample[value_col] = corrected_values

    return df_sample
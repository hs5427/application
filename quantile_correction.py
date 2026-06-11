import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

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
    """
    生成サンプルのvalueを、元データの分位点カーブに合わせて補正する。

    処理:
    1. サンプル値 x から u = F_model(x|q) を計算
    2. 元データ側の q における PPF_ref(u|q) を補間
    3. value を補正値に置換
    """

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

# def apply_quantile_correction(
#     df_sample,
#     df_reference,
#     q_col="TTP_Percentile",
#     p_col="value_Percentile",
#     value_col="value"
# ):
#     """
#     サンプル値を元データの分位点に合わせて補正
#     """

#     df_sample = df_sample.copy()

#     corrected_values = []

#     for q, df_group in df_sample.groupby(q_col):

#         df_ref_same_q = df_reference[
#             df_reference[q_col] == q
#         ]

#         if len(df_ref_same_q) < 2:
#             corrected_values.extend(
#                 df_group[value_col].tolist()
#             )
#             continue

#         x_ref = df_ref_same_q[value_col].to_numpy(dtype=float)
#         p_ref = df_ref_same_q[p_col].to_numpy(dtype=float)

#         sort_idx = np.argsort(x_ref)

#         x_ref = x_ref[sort_idx]
#         p_ref = p_ref[sort_idx]

#         cdf_interp = PchipInterpolator(
#             x_ref,
#             p_ref,
#             extrapolate=True
#         )

#         p_sample = np.clip(
#             cdf_interp(
#                 df_group[value_col].to_numpy(dtype=float)
#             ),
#             p_ref.min(),
#             p_ref.max()
#         )

#         ppf_interp = PchipInterpolator(
#             p_ref,
#             x_ref,
#             extrapolate=True
#         )

#         corrected = ppf_interp(p_sample)

#         corrected_values.extend(corrected)

#     df_sample[value_col] = corrected_values

#     return df_sample
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

from mixed_weibull import (
    fit_by_throughput_percentile,
    fit_independent_mixture_weibull,
    build_global_mixture_weibull_cdf,
    build_independent_mixture_weibull_cdf,
    make_weibull_param_coefficients,
    plot_weibull_parameter_trends,
)

from mixed_gaussian import (
    fit_gaussian_by_throughput_percentile,
    fit_independent_mixture_gaussian,
    build_global_mixture_gaussian_cdf,
    build_independent_mixture_gaussian_cdf,
    make_gaussian_param_coefficients,
    plot_gaussian_parameter_trends
)

from spline_model import (
    fit_independent_spline_model,
    fit_dependent_spline_model,
    build_spline_sampler
)

from sampling import (
    build_bivariate_sampler,
    build_independent_sampler
)

from validation import (
    make_qq_table_by_ttp,
    plot_qq_by_ttp_grid,
    plot_independent_qq,
    make_independent_qq_table,
    make_validation_metrics_from_qq,
    plot_sample_distribution_overlay,
    plot_sample_pdf_histogram,
    plot_cdf_validation_grid,
    plot_cdf_validation
)

from quantile_correction import (
    apply_quantile_correction,
    apply_independent_quantile_correction
)

# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Fleet Data Analysis App",
    layout="wide"
)

st.title("Fleet Data Analysis App")


# =========================
# Session state initialize
# =========================

default_keys = {
    # Tab1用
    "fleetdata_path": "/Users/hisashi/Desktop/data/data.xlsx",
    "throughput_quantile_path": "/Users/hisashi/Desktop/data/Percentile.xlsx",
    "fleetdata_sheet_names": [],
    "selected_feature": None,
    "df_fleetdata": None,
    "df_long": None,
    "df_throughput_quantile": None,
    "analysis_mode": "TTP dependent",

    # Tab3用
    "model_type": "混合ワイブルモデル",
    "selected_n_components": 2,
    "use_quantile_correction": False,
    "df_fit_results_dict": {},
    "global_cdf": None,
    "global_ppf": None,
    "df_weibull_poly_coefficients": None,
    "df_weibull_param_coefficients": None,

    # Tab4用
    "use_quantile_correction": False,

    # 後続用
    "n_samples": 10_000,
    "random_state": 42,
    "df_sample": None,
    "df_corrected": None,
    "df_validation": None,
    "df_validation_metrics": None,
    "hist_bins": 50,

    "param_fit_method_ui": "3次スプライン補完",
    "df_param_coefficients": None,
    "fig_param_trends": None,
}

for key, value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================
# Tabs
# =========================

tab_data, tab_viz, tab_fit, tab_sample, tab_valid, tab_export = st.tabs([
    "1. Data",
    "2. Visualization",
    "3. Fitting",
    "4. Sampling",
    "5. Validation",
    "6. Export"
])

# =========================
# Tab 1: Data
# =========================

with tab_data:
    st.header("1. Data Input")
    st.divider()

    # =========================
    # 1-1. File path input
    # =========================

    st.subheader("Input files")

    col1, col2 = st.columns(2)

    with col1:

        fleetdata_path = st.text_input(
            "Fleet Data file (Excel file)",
            value=st.session_state["fleetdata_path"],
            placeholder=r"C:\Data\fleetdata.xlsx"
        )

        st.session_state["fleetdata_path"] = fleetdata_path

        if st.button("Check Fleet Data file", key="check_fleetdata_file"):
            try:
                path = Path(fleetdata_path)
                if not path.exists():
                    st.error("Fleet data file does not exist.")
                elif path.suffix.lower() not in [".xlsx", ".xls"]:
                    st.error("Please select an Excel file.")
                else:
                    xls = pd.ExcelFile(fleetdata_path)
                    st.session_state["fleetdata_sheet_names"] = xls.sheet_names
                    st.success("Fleet data file checked successfully.")
            except Exception as e:
                st.error(f"Failed to check fleet data file: {e}")

    with col2:

        throughput_quantile_path = st.text_input(
            "Total Throughput data (Excel file)",
            value=st.session_state["throughput_quantile_path"],
            placeholder=r"C:\Data\throughput_quantile.xlsx"
        )

        st.session_state["throughput_quantile_path"] = throughput_quantile_path

        if st.button("Check Total Throughput file", key="check_ttp_file"):
            try:
                path = Path(throughput_quantile_path)
                if not path.exists():
                    st.warning(
                        "Total Throughput quantile file does not exist. "
                        "Independent data can still be loaded without this file."
                    )
                elif path.suffix.lower() not in [".xlsx", ".xls"]:
                    st.error("Please select an Excel file.")
                else:
                    st.success("Total Throughput quantile file checked successfully.")
            except Exception as e:
                st.error(f"Failed to check Total Throughput quantile file: {e}")

    st.divider()

    # =========================
    # 1-2. Feature sheet selection
    # =========================

    st.subheader("Feature selection")

    sheet_names = st.session_state["fleetdata_sheet_names"]

    if len(sheet_names) == 0:
        st.info("Fleet Data file を指定し、'Check Fleet Data file' を押してください。")
    else:
        selected_feature = st.selectbox(
            "解析したい特徴量",
            options=sheet_names,
            index=0
        )

        st.session_state["selected_feature"] = selected_feature
        # st.write(f"選択中の解析対象: **{selected_feature}**")
        st.divider()

        # =========================
        # 1-3. Load data
        # =========================

        st.subheader("Load Data")

        if st.button("Load Data", key="load_data"):
            try:
                # =========================
                # df_fleetdata
                # =========================

                df_fleetdata = pd.read_excel(
                    st.session_state["fleetdata_path"],
                    sheet_name=selected_feature,
                    index_col=0
                )

                st.session_state["df_fleetdata"] = df_fleetdata

                # =========================
                # Auto detect analysis mode
                # =========================

                if len(df_fleetdata) == 1:
                    analysis_mode = "Independent"
                else:
                    analysis_mode = "TTP dependent"

                st.session_state["analysis_mode"] = analysis_mode
                # st.info(f"Detected analysis mode: {analysis_mode}")

                # =========================
                # Long format
                # =========================

                if analysis_mode == "TTP dependent":

                    df_fleetdata.index.name = "TTP_Percentile"
                    df_fleetdata.columns.name = "value_Percentile"

                    df_long = (
                        df_fleetdata
                        .stack()
                        .reset_index(name="value")
                    )

                    df_long.columns = [
                        "TTP_Percentile",
                        "value_Percentile",
                        "value"
                    ]

                    # =========================
                    # df_throughput_quantile
                    # =========================

                    throughput_path = st.session_state["throughput_quantile_path"]

                    if throughput_path is None or throughput_path == "":
                        raise ValueError(
                            "TTP dependent data requires Total Throughput data file."
                        )

                    path = Path(throughput_path)

                    if not path.exists():
                        raise FileNotFoundError(
                            "TTP dependent data requires valid Total Throughput data file."
                        )

                    df_throughput_quantile = pd.read_excel(
                        throughput_path
                    )

                    df_throughput_quantile.columns = [
                        "TTP_Percentile",
                        "TTP"
                    ]

                    st.session_state["df_throughput_quantile"] = df_throughput_quantile

                else:

                    df_fleetdata_one_row = df_fleetdata.iloc[[0]].copy()

                    df_long = (
                        df_fleetdata_one_row
                        .T
                        .reset_index()
                    )

                    df_long.columns = [
                        "value_Percentile",
                        "value"
                    ]

                    df_long["value_Percentile"] = df_long["value_Percentile"].astype(float)
                    df_long["value"] = df_long["value"].astype(float)

                    # TTPに依存しないため、Throughputデータは使わない
                    st.session_state["df_throughput_quantile"] = None

                st.session_state["df_long"] = df_long

                # =========================
                # Reset downstream results
                # =========================

                st.session_state["df_fit_results_dict"] = {}
                st.session_state["global_cdf"] = None

                st.session_state["df_sample"] = None
                st.session_state["df_corrected"] = None

                st.session_state["df_validation"] = None
                st.session_state["df_validation_metrics"] = None

                st.session_state["df_weibull_param_coefficients"] = None
                st.session_state["fig_weibull_param_trends"] = None

                st.success("Data loaded successfully.")

            except Exception as e:
                st.error(f"Failed to load data: {e}")

    st.divider()

    # =========================
    # 1-4. Data Preview
    # =========================

    st.subheader("Data Preview")

    st.write("df_fleetdata")

    if st.session_state["df_fleetdata"] is not None:
        st.dataframe(
            st.session_state["df_fleetdata"],
            use_container_width=True
        )
    else:
        st.info("df_fleetdata is not loaded yet.")

    st.write("df_throughput_quantile")

    if st.session_state.get("analysis_mode") == "TTP dependent":

        if st.session_state["df_throughput_quantile"] is not None:
            st.dataframe(
                st.session_state["df_throughput_quantile"],
                use_container_width=True
            )
        else:
            st.info("df_throughput_quantile is not loaded yet.")

    elif st.session_state.get("analysis_mode") == "Independent":
        st.info("Independent modeでは df_throughput_quantile は使用しません。")

    else:
        st.info("Analysis mode is not detected yet.")
        

# =========================
# Tab 2: Visualization
# =========================

with tab_viz:

    st.header("2. Data Visualization")

    df_long = st.session_state["df_long"]

    selected_feature = st.session_state.get(
        "selected_feature",
        "Value"
    )

    if df_long is None:
        st.info("Please load data first.")
    else:
        analysis_mode = st.session_state.get(
            "analysis_mode",
            "TTP dependent"
        )
        color_col = (
            "TTP_Percentile"
            if "TTP_Percentile" in df_long.columns
            else None
        )

        # =========================
        # Plot settings
        # =========================

        col1, col2 = st.columns([1, 3])

        with col1:

            # st.subheader("Plot settings")

            st.markdown("#### Graph size")

            plot_height = st.slider(
                "Plot height",
                min_value=400,
                max_value=1000,
                value=600,
                step=50
            )

            x_min_data = float(df_long["value"].min())
            x_max_data = float(df_long["value"].max())

            st.markdown("#### X-axis range")

            col_min, col_max = st.columns(2)

            with col_min:
                x_min = st.number_input(
                    "X min",
                    value=x_min_data
                )

            with col_max:
                x_max = st.number_input(
                    "X max",
                    value=x_max_data
                )

        # =========================
        # Plot
        # =========================

        with col2:

            fig = px.line(
                df_long,
                x="value",
                y="value_Percentile",
                color=color_col,
                markers=True
            )

            fig.update_layout(
                height=plot_height,
                xaxis_title=selected_feature,
                yaxis_title="Cumulative Probability",
                legend_title="TTP_Percentile" if color_col else None,
                showlegend=True if color_col else False
            )

            fig.update_xaxes(
                range=[x_min, x_max]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================
# Tab 3: Fitting
# =========================

with tab_fit:
    st.header("3. Model Fitting")
    st.divider()

    df_long = st.session_state["df_long"]

    # =========================
    # 3-1. Model selection
    # =========================

    st.subheader("Model selection")

    model_type = st.selectbox(
        "モデルを選択",
        options=[
            "混合ワイブルモデル",
            "混合正規分布モデル",
            "スプライン補完モデル",
        ],
        index=[
            "混合ワイブルモデル",
            "混合正規分布モデル",
            "スプライン補完モデル",
        ].index(st.session_state["model_type"])
        if st.session_state["model_type"] in [
            "混合ワイブルモデル",
            "混合正規分布モデル",
            "スプライン補完モデル",
        ]
        else 0
    )

    st.session_state["model_type"] = model_type

    st.divider()

    # =========================
    # 3-2. Model-specific settings
    # =========================

    st.subheader("Model settings")

    if model_type in ["混合ワイブルモデル", "混合正規分布モデル"]:

        col1, col2 = st.columns([1, 1])

        # モデルの項数を取得
        with col1:
            current_n_components = st.session_state.get("selected_n_components", 2)

            if current_n_components not in [1, 2, 3]:
                current_n_components = 2

            selected_n_components = st.selectbox(
                "混合モデルの項数",
                options=[1, 2, 3],
                index=[1, 2, 3].index(current_n_components)
            )

            st.session_state["selected_n_components"] = selected_n_components

        # TTP方向の補完方法を取得
        with col2:
            param_fit_method_ui = st.selectbox(
                "TTP方向のパラメータ補完方法",
                options=[
                    "3次スプライン補完",
                    "3次多項式"
                ],
                index=[
                    "3次スプライン補完",
                    "3次多項式"
                ].index(
                    st.session_state.get(
                        "param_fit_method_ui",
                        "3次スプライン補完"
                    )
                )
            )

            st.session_state["param_fit_method_ui"] = param_fit_method_ui
            # st.session_state["weibull_param_fit_method"] = param_fit_method_ui

        st.info(
            f"選択中: {model_type} / "
            f"{selected_n_components}項 / "
            f"TTP方向: {param_fit_method_ui}"
        )

    elif model_type == "スプライン補完モデル":

        st.info(
            "スプライン補完では、項数とTTP方向の補完方法の設定は不要です。"
        )

        st.session_state["use_quantile_correction"] = False

    st.divider()

    # =========================
    # Fitting execution
    # =========================

    if df_long is None:
        st.info("Please load data first.")

    else:
        st.subheader("Run fitting")

        if st.button("Run fitting", key="run_fitting"):

            try:
                model_type = st.session_state["model_type"]

                # =========================
                # 混合ワイブルモデル
                # =========================

                if model_type == "混合ワイブルモデル":

                    analysis_mode = st.session_state.get(
                        "analysis_mode",
                        "TTP dependent"
                    )

                    n_components = st.session_state["selected_n_components"]

                    method_map = {
                        "3次スプライン補完": "spline",
                        "3次多項式": "poly3"
                    }

                    method_ui = st.session_state["param_fit_method_ui"]
                    param_fit_method = method_map[method_ui]

                    # =========================
                    # TTP dependent
                    # =========================

                    if analysis_mode == "TTP dependent":

                        # 各TTPパーセンタイルごとにFitting
                        df_fit = fit_by_throughput_percentile(
                            df_long,
                            n_components=n_components
                        )

                        df_fit_results_dict = {
                            n_components: df_fit
                        }

                        st.session_state["df_fit_results_dict"] = df_fit_results_dict

                        # TTP方向にパラメータを補完
                        if df_fit.empty:
                            st.session_state["global_cdf"] = None
                            st.session_state["df_weibull_param_coefficients"] = None

                            st.warning(
                                f"混合ワイブルモデル {n_components}成分のフィッティング結果がありません。"
                            )

                        else:
                            global_cdf = build_global_mixture_weibull_cdf(
                                df_fit_results_dict=df_fit_results_dict,
                                n_components=n_components,
                                q_col="TTP_Percentile",
                                param_fit_method=param_fit_method
                            )

                            st.session_state["global_cdf"] = global_cdf

                            df_coef = make_weibull_param_coefficients(
                                df_fit=df_fit,
                                n_components=n_components,
                                q_col="TTP_Percentile",
                                method=param_fit_method
                            )

                            st.session_state["df_weibull_param_coefficients"] = df_coef

                            fig_param = plot_weibull_parameter_trends(
                                df_fit=df_fit,
                                n_components=n_components,
                                q_col="TTP_Percentile",
                                param_fit_method=param_fit_method
                            )

                            st.session_state["fig_weibull_param_trends"] = fig_param

                            st.success(
                                f"混合ワイブルモデル {n_components}成分のフィッティングとGlobal model作成が完了しました。"
                            )

                    # =========================
                    # Independent
                    # =========================

                    else:

                        df_fit = fit_independent_mixture_weibull(
                            df_long,
                            n_components=n_components,
                            p_col="value_Percentile",
                            value_col="value"
                        )

                        df_fit_results_dict = {
                            n_components: df_fit
                        }

                        st.session_state["df_fit_results_dict"] = df_fit_results_dict

                        if df_fit.empty:
                            st.session_state["global_cdf"] = None
                            st.session_state["df_weibull_param_coefficients"] = None

                            st.warning(
                                f"Independent 混合ワイブルモデル {n_components}成分のフィッティング結果がありません。"
                            )

                        else:
                            global_cdf = build_independent_mixture_weibull_cdf(
                                df_fit=df_fit,
                                n_components=n_components
                            )

                            st.session_state["global_cdf"] = global_cdf

                            # IndependentではTTP方向の係数・パラメータ推移は作らない
                            st.session_state["df_weibull_param_coefficients"] = None
                            st.session_state["fig_weibull_param_trends"] = None

                            st.success(
                                f"Independent 混合ワイブルモデル {n_components}成分のフィッティングが完了しました。"
                            )

                # =========================
                # 混合正規分布モデル
                # =========================

                elif model_type == "混合正規分布モデル":

                    analysis_mode = st.session_state.get(
                        "analysis_mode",
                        "TTP dependent"
                    )

                    n_components = st.session_state["selected_n_components"]

                    method_map = {
                        "3次スプライン補完": "spline",
                        "3次多項式": "poly3"
                    }

                    method_ui = st.session_state.get(
                        "param_fit_method_ui",
                        st.session_state.get(
                            "param_fit_method_ui"
                            "3次スプライン補完"
                        )
                    )

                    param_fit_method = method_map[method_ui]

                    # 従属パラメータ
                    if analysis_mode == "TTP dependent":

                        df_fit = fit_gaussian_by_throughput_percentile(
                            df_long,
                            n_components=n_components
                        )

                        df_fit_results_dict = {
                            n_components: df_fit
                        }

                        st.session_state["df_fit_results_dict"] = df_fit_results_dict

                        if df_fit.empty:
                            st.session_state["global_cdf"] = None
                            st.session_state["df_param_coefficients"] = None
                            st.session_state["fig_param_trends"] = None

                            st.warning(
                                f"混合正規分布モデル {n_components}成分のフィッティング結果がありません。"
                            )

                        else:
                            global_cdf = build_global_mixture_gaussian_cdf(
                                df_fit_results_dict=df_fit_results_dict,
                                n_components=n_components,
                                q_col="TTP_Percentile",
                                param_fit_method=param_fit_method
                            )

                            st.session_state["global_cdf"] = global_cdf

                            df_coef = make_gaussian_param_coefficients(
                                df_fit=df_fit,
                                n_components=n_components,
                                q_col="TTP_Percentile",
                                method=param_fit_method
                            )

                            st.session_state["df_param_coefficients"] = df_coef
                            st.session_state["df_weibull_param_coefficients"] = None

                            fig_param = plot_gaussian_parameter_trends(
                                df_fit=df_fit,
                                n_components=n_components,
                                q_col="TTP_Percentile",
                                param_fit_method=param_fit_method
                            )

                            st.session_state["fig_param_trends"] = fig_param
                            st.session_state["fig_weibull_param_trends"] = None

                            st.success(
                                f"混合正規分布モデル {n_components}成分のフィッティングとGlobal model作成が完了しました。"
                            )

                    # 非従属パラメータ
                    else:
                        df_fit = fit_independent_mixture_gaussian(
                            df_long=df_long,
                            n_components=n_components,
                            p_col="value_Percentile",
                            value_col="value"
                        )

                        df_fit_results_dict = {
                            n_components: df_fit
                        }

                        st.session_state["df_fit_results_dict"] = df_fit_results_dict

                        if df_fit.empty:
                            st.session_state["global_cdf"] = None
                            st.session_state["df_param_coefficients"] = None
                            st.session_state["fig_param_trends"] = None

                            st.warning(
                                f"Independent 混合正規分布モデル {n_components}成分のフィッティング結果がありません。"
                            )

                        else:
                            global_cdf = build_independent_mixture_gaussian_cdf(
                                df_fit=df_fit,
                                n_components=n_components
                            )

                            st.session_state["global_cdf"] = global_cdf

                            st.session_state["df_param_coefficients"] = None
                            st.session_state["df_weibull_param_coefficients"] = None
                            st.session_state["fig_param_trends"] = None
                            st.session_state["fig_weibull_param_trends"] = None

                            st.success(
                                f"Independent 混合正規分布モデル {n_components}成分のフィッティングが完了しました。"
                            )

                # =========================
                # スプライン補完モデル
                # =========================
                elif model_type == "スプライン補完モデル":

                    analysis_mode = st.session_state.get(
                        "analysis_mode",
                        "TTP dependent"
                    )

                    if analysis_mode == "TTP dependent":

                        result_spline = fit_dependent_spline_model(
                            df_long=df_long,
                            q_col="TTP_Percentile",
                            p_col="value_Percentile",
                            value_col="value"
                        )

                    else:

                        result_spline = fit_independent_spline_model(
                            df_long=df_long,
                            p_col="value_Percentile",
                            value_col="value"
                        )

                    df_fit = result_spline["df_fit"]
                    global_cdf = result_spline["global_cdf"]
                    global_ppf = result_spline["global_ppf"]

                    st.session_state["df_fit_results_dict"] = {
                        1: df_fit
                    }

                    st.session_state["global_cdf"] = global_cdf
                    st.session_state["global_ppf"] = global_ppf

                    st.session_state["df_param_coefficients"] = None
                    st.session_state["df_weibull_param_coefficients"] = None
                    st.session_state["fig_param_trends"] = None
                    st.session_state["fig_weibull_param_trends"] = None

                    st.success("スプライン補完モデルの作成が完了しました。")

            except Exception as e:
                st.session_state["global_cdf"] = None
                st.error(f"Fitting failed: {e}")

        st.divider()

    # =========================
    # Fit results
    # =========================

    st.subheader("Fit results")

    if st.session_state["df_fit_results_dict"]:
        for n, df_fit in st.session_state["df_fit_results_dict"].items():
            if df_fit is not None and not df_fit.empty:
                st.dataframe(df_fit, use_container_width=True)
            else:
                st.info("No fit result.")
    else:
        st.info("No fitting result yet.")
    
    st.subheader("Parameter coefficients")

    model_type = st.session_state.get("model_type")

    if model_type == "混合ワイブルモデル":
        df_coef = st.session_state.get("df_weibull_param_coefficients")
    else:
        df_coef = st.session_state.get("df_param_coefficients")

    if df_coef is not None and not df_coef.empty:
        st.dataframe(
            df_coef,
            use_container_width=True
        )
    else:
        st.info("No parameter coefficient result yet.")


    st.subheader("Parameter trends")

    analysis_mode = st.session_state.get("analysis_mode", "TTP dependent")
    model_type = st.session_state.get("model_type")

    if analysis_mode != "TTP dependent":
        st.info("Independent modeではTTP方向のパラメータ推移は表示しません。")

    elif st.session_state["df_fit_results_dict"]:

        n_components = st.session_state["selected_n_components"]
        df_fit = st.session_state["df_fit_results_dict"].get(n_components)

        if df_fit is not None and not df_fit.empty:

            if "TTP_Percentile" not in df_fit.columns:
                st.info("TTP_Percentile列がないため、パラメータ推移は表示しません。")

            else:
                method_map = {
                    "3次スプライン補完": "spline",
                    "3次多項式": "poly3"
                }

                method_ui = st.session_state.get(
                    "param_fit_method_ui",
                    st.session_state.get(
                        "param_fit_method_ui",
                        "3次スプライン補完"
                    )
                )

                param_fit_method = method_map[method_ui]

                if model_type == "混合ワイブルモデル":
                    fig_param = plot_weibull_parameter_trends(
                        df_fit=df_fit,
                        n_components=n_components,
                        q_col="TTP_Percentile",
                        param_fit_method=param_fit_method
                    )

                elif model_type == "混合正規分布モデル":
                    fig_param = plot_gaussian_parameter_trends(
                        df_fit=df_fit,
                        n_components=n_components,
                        q_col="TTP_Percentile",
                        param_fit_method=param_fit_method
                    )

                else:
                    fig_param = None

                if fig_param is not None:
                    st.plotly_chart(
                        fig_param,
                        use_container_width=True
                    )
                else:
                    st.info("No parameter trend plot available.")

        else:
            st.info("No fit result for selected component.")

    else:
        st.info("No fitting result yet.")


# =========================
# Tab 4: Sampling
# =========================

with tab_sample:
    st.header("4. Random Sampling")
    st.divider()

    selected_feature = st.session_state.get("selected_feature", "value")
    analysis_mode = st.session_state.get("analysis_mode", "TTP dependent")
    model_type = st.session_state.get("model_type", "混合ワイブルモデル")

    if model_type == "混合ワイブルモデル":
        model_family = "weibull"
    elif model_type == "混合正規分布モデル":
        model_family = "gaussian"
    else:
        model_family = "weibull"

    # =========================
    # Sampling settings
    # =========================

    st.subheader("Sampling settings")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        n_samples = st.number_input(
            "Number of random samples",
            min_value=100,
            max_value=1_000_000,
            value=st.session_state["n_samples"],
            step=1000
        )

    with col2:
        random_state = st.number_input(
            "Random seed",
            value=st.session_state["random_state"],
            step=1
        )

    with col3:
        use_quantile_correction = st.checkbox(
            "Apply quantile correction",
            value=st.session_state.get(
                "use_quantile_correction",
                False
            )
        )

    st.session_state["n_samples"] = n_samples
    st.session_state["random_state"] = random_state
    st.session_state["use_quantile_correction"] = use_quantile_correction

    st.info(f"Analysis mode: {analysis_mode}")
    st.info(f"Model type: {model_type}")

    st.divider()

    # =========================
    # Pre-check
    # =========================
    st.subheader("Generated sample visualization")

    if st.session_state["global_cdf"] is None:
        st.info("Please run fitting first.")

    elif (
        analysis_mode == "TTP dependent"
        and st.session_state["df_throughput_quantile"] is None
    ):
        st.info("Please load Total Throughput quantile data first.")

    else:
        if st.button("Generate random samples", key="generate_random_samples"):
            try:

                # =========================
                # Sampling
                # =========================

                model_type = st.session_state.get(
                    "model_type",
                    "混合ワイブルモデル"
                )

                if model_type == "スプライン補完モデル":

                    sampler = build_spline_sampler(
                        global_ppf=st.session_state["global_ppf"],
                        analysis_mode=analysis_mode,
                        df_throughput_quantile=st.session_state.get("df_throughput_quantile"),
                        q_col="TTP_Percentile",
                        throughput_col="TTP"
                    )

                    df_sample = sampler(
                        n_samples=st.session_state["n_samples"],
                        random_state=st.session_state["random_state"]
                    )

                else:

                    if analysis_mode == "TTP dependent":
                        sampler = build_bivariate_sampler(
                            df_throughput_quantile=st.session_state["df_throughput_quantile"],
                            global_cdf=st.session_state["global_cdf"],
                            q_col="TTP_Percentile",
                            throughput_col="TTP",
                            model_family=model_family
                        )

                        df_sample = sampler(
                            n_samples=st.session_state["n_samples"],
                            random_state=st.session_state["random_state"]
                        )

                    else:
                        sampler = build_independent_sampler(
                            global_cdf=st.session_state["global_cdf"],
                            model_family=model_family
                        )

                        df_sample = sampler(
                            n_samples=st.session_state["n_samples"],
                            random_state=st.session_state["random_state"]
                        )

                # =========================
                # Quantile correction
                # =========================

                if st.session_state["use_quantile_correction"]:

                    if analysis_mode == "TTP dependent":
                        df_sample = apply_quantile_correction(
                            df_sample=df_sample,
                            df_reference=st.session_state["df_long"],
                            global_cdf=st.session_state["global_cdf"],
                            q_col="TTP_Percentile",
                            p_col="value_Percentile",
                            value_col="value"
                        )

                    else:
                        df_sample = apply_independent_quantile_correction(
                            df_sample=df_sample,
                            df_reference=st.session_state["df_long"],
                            global_cdf=st.session_state["global_cdf"],
                            p_col="value_Percentile",
                            value_col="value"
                        )

                st.session_state["df_sample"] = df_sample

                if "value_raw" in df_sample.columns:
                    st.write("Raw mean:", df_sample["value_raw"].mean())
                    st.write("Corrected mean:", df_sample["value"].mean())

                st.success("Random samples generated successfully.")

            except Exception as e:
                st.error(f"Failed to generate random samples: {e}")
        
        st.divider()

        # =========================
        # Visualization
        # =========================

        if st.session_state["df_sample"] is not None:
            df_sample = st.session_state["df_sample"]

            if not df_sample.empty:

                value_min = float(df_sample["value"].min())
                value_max = float(df_sample["value"].max())

                # =========================
                # PDF + Histogram
                # =========================

                st.markdown("### PDF + Histogram")

                pdf_hist_bins = st.number_input(
                    "PDF histogram bins",
                    min_value=5,
                    max_value=300,
                    value=50,
                    step=5,
                    key="sampling_pdf_hist_bins"
                )

                fig_pdf_hist = plot_sample_pdf_histogram(
                    df_sample=df_sample,
                    selected_feature=selected_feature,
                    value_col="value",
                    bins=pdf_hist_bins
                )

                if fig_pdf_hist is not None:
                    st.plotly_chart(
                        fig_pdf_hist,
                        use_container_width=True
                    )
                else:
                    st.info("No PDF + histogram plot available.")

                # =========================
                # Scatter
                # =========================

                if analysis_mode == "TTP dependent" and "TTP" in df_sample.columns:

                    st.markdown("### Scatter plot")

                    ttp_min = float(df_sample["TTP"].min())
                    ttp_max = float(df_sample["TTP"].max())

                    col_scatter_setting, col_scatter_plot = st.columns([1, 3])

                    with col_scatter_setting:
                        scatter_x_min = st.number_input(
                            "Scatter X min",
                            value=ttp_min,
                            key="scatter_x_min"
                        )

                        scatter_x_max = st.number_input(
                            "Scatter X max",
                            value=ttp_max,
                            key="scatter_x_max"
                        )

                        scatter_y_min = st.number_input(
                            "Scatter Y min",
                            value=value_min,
                            key="scatter_y_min"
                        )

                        scatter_y_max = st.number_input(
                            "Scatter Y max",
                            value=value_max,
                            key="scatter_y_max"
                        )

                    with col_scatter_plot:
                        fig_scatter = px.scatter(
                            df_sample,
                            x="TTP",
                            y="value",
                            opacity=0.35
                        )

                        fig_scatter.update_layout(
                            height=500,
                            xaxis_title="Total Throughput",
                            yaxis_title=selected_feature,
                            showlegend=False
                        )

                        fig_scatter.update_xaxes(
                            range=[scatter_x_min, scatter_x_max]
                        )

                        fig_scatter.update_yaxes(
                            range=[scatter_y_min, scatter_y_max]
                        )

                        st.plotly_chart(
                            fig_scatter,
                            use_container_width=True
                        )

                else:
                    st.info("Independent modeでは散布図は表示しません。")

                # =========================
                # Summary
                # =========================

                st.subheader("Generated sample summary")

                st.dataframe(
                    df_sample.describe(),
                    use_container_width=True
                )


# =========================
# Tab 5: Validation
# =========================

with tab_valid:

    st.header("5. Validation")
    st.divider()

    df_long = st.session_state["df_long"]
    df_sample = st.session_state["df_sample"]
    selected_feature = st.session_state.get("selected_feature", "value")
    analysis_mode = st.session_state.get("analysis_mode", "TTP dependent")

    if df_long is None:
        st.info("Please load data first.")

    elif df_sample is None or df_sample.empty:
        st.info("Please generate random samples first.")

    else:
        st.subheader("Validation settings")

        st.info(f"Analysis mode: {analysis_mode}")

        target_percentiles = [
            0.01, 0.25, 0.50, 0.75,
            0.90, 0.95, 0.99, 0.998
        ]

        run_validation = st.button(
            "Run validation",
            key="run_validation"
        )

        if run_validation:
            try:
                if analysis_mode == "TTP dependent":
                    df_qq = make_qq_table_by_ttp(
                        df_long=df_long,
                        df_sample=df_sample,
                        percentiles=target_percentiles,
                        q_col="TTP_Percentile",
                        p_col="value_Percentile",
                        value_col="value",
                        ttp_bin_width=2.5
                    )

                else:
                    df_qq = make_independent_qq_table(
                        df_long=df_long,
                        df_sample=df_sample,
                        percentiles=target_percentiles,
                        p_col="value_Percentile",
                        value_col="value"
                    )

                st.session_state["df_validation"] = df_qq

                df_metrics = make_validation_metrics_from_qq(df_qq)
                st.session_state["df_validation_metrics"] = df_metrics

                if df_qq is None or df_qq.empty:
                    st.warning(
                        "Q-Q table is empty. Check input data and percentile values."
                    )
                else:
                    st.success("Validation completed.")

            except Exception as e:
                st.session_state["df_validation"] = None
                st.session_state["df_validation_metrics"] = None
                st.error(f"Validation failed: {e}")

        st.divider()

        # =========================
        # Metrics
        # =========================

        st.subheader("Validation metrics")

        df_metrics = st.session_state.get("df_validation_metrics")

        if df_metrics is not None and not df_metrics.empty:

            metric_dict = dict(
                zip(
                    df_metrics["Metric"],
                    df_metrics["Value"]
                )
            )

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                qq_r2 = metric_dict.get("QQ_R2_corr", np.nan)
                st.metric(
                    label="QQ R²",
                    value=f"{qq_r2:.4f}" if pd.notna(qq_r2) else "NaN"
                )

            with col_m2:
                slope = metric_dict.get("QQ_slope", np.nan)
                st.metric(
                    label="QQ slope",
                    value=f"{slope:.4f}" if pd.notna(slope) else "NaN"
                )

            with col_m3:
                rmse = metric_dict.get("RMSE", np.nan)
                st.metric(
                    label="RMSE",
                    value=f"{rmse:.4g}" if pd.notna(rmse) else "NaN"
                )

            with col_m4:
                mae = metric_dict.get("MAE", np.nan)
                st.metric(
                    label="MAE",
                    value=f"{mae:.4g}" if pd.notna(mae) else "NaN"
                )

            st.dataframe(
                df_metrics,
                use_container_width=True
            )

        else:
            st.info("No validation metrics yet.")

        st.divider()

        # =========================
        # CDF validation
        # =========================

        st.subheader("CDF validation")

        try:
            if analysis_mode == "TTP dependent":

                fig_cdf = plot_cdf_validation_grid(
                    df_sample=df_sample,
                    df_reference=df_long,
                    selected_feature=selected_feature,
                    value_col="value",
                    p_col="value_Percentile",
                    q_col="TTP_Percentile",
                    ttp_bin_width=0.025
                )

            else:

                fig_cdf = plot_cdf_validation(
                    df_sample=df_sample,
                    df_reference=df_long,
                    selected_feature=selected_feature,
                    value_col="value",
                    p_col="value_Percentile"
                )

            if fig_cdf is not None:
                st.plotly_chart(
                    fig_cdf,
                    use_container_width=True
                )
            else:
                st.info("No CDF validation plot available.")

        except Exception as e:
            st.error(f"Failed to plot CDF validation: {e}")

        st.divider()

        # =========================
        # Q-Q Plot
        # =========================

        st.subheader("Q-Q plot")

        df_qq = st.session_state.get("df_validation")

        if df_qq is not None and not df_qq.empty:
            try:
                if analysis_mode == "TTP dependent":
                    fig_qq = plot_qq_by_ttp_grid(
                        df_qq=df_qq,
                        selected_feature=selected_feature,
                        q_col="TTP_Percentile",
                        p_col="value_Percentile",
                        n_cols=4
                    )

                else:
                    fig_qq = plot_independent_qq(
                        df_qq=df_qq,
                        selected_feature=selected_feature
                    )

                if fig_qq is not None:
                    st.plotly_chart(
                        fig_qq,
                        use_container_width=True
                    )
                else:
                    st.info("No Q-Q plot available.")

            except Exception as e:
                st.error(f"Failed to plot Q-Q plot: {e}")

            st.divider()

            st.subheader("Q-Q table")

            st.dataframe(
                df_qq,
                use_container_width=True
            )

        else:
            st.info("No Q-Q result yet.")
            

# =========================
# Tab 6: Export
# =========================

with tab_export:
    st.header("6. Export")

    st.subheader("Export settings")

    file_prefix = st.text_input(
        "Output file name",
        value="fleetdata_analysis_result",
        placeholder="例: DOD_mixed_weibull_result"
    )

    export_sample = st.checkbox(
        "Export sampling data",
        value=True
    )

    export_fit_params = st.checkbox(
        "Export fitting parameters",
        value=True
    )

    export_qq = st.checkbox(
        "Export Q-Q plot data",
        value=True
    )

    export_param_coefficients = st.checkbox(
        "Parameter coefficients",
        value=False,
        key="export_param_coefficients"
    )

    st.divider()

    # =========================
    # Sampling data
    # =========================

    if export_sample:
        st.subheader("Sampling data")

        df_sample = st.session_state.get("df_sample")

        if df_sample is not None and not df_sample.empty:
            csv_sample = df_sample.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download sampling data CSV",
                data=csv_sample,
                file_name=f"{file_prefix}_sampling_data.csv",
                mime="text/csv",
                key="download_sampling_data"
            )
        else:
            st.info("No sampling data available.")

    # =========================
    # Fitting parameters
    # =========================

    if export_fit_params:
        st.subheader("Fitting parameters")

        df_fit_results_dict = st.session_state.get("df_fit_results_dict", {})

        if df_fit_results_dict:
            for n_components, df_fit in df_fit_results_dict.items():
                if df_fit is not None and not df_fit.empty:
                    csv_fit = df_fit.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label=f"Download fitting parameters CSV ({n_components} components)",
                        data=csv_fit,
                        file_name=f"{file_prefix}_fit_params_{n_components}components.csv",
                        mime="text/csv",
                        key=f"download_fit_params_{n_components}"
                    )
                else:
                    st.info(f"No fitting parameters for {n_components} components.")
        else:
            st.info("No fitting parameter data available.")

    # =========================
    # Q-Q plot data
    # =========================

    if export_qq:
        st.subheader("Q-Q plot data")

        df_qq = st.session_state.get("df_validation")

        if df_qq is not None and not df_qq.empty:
            csv_qq = df_qq.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download Q-Q plot data CSV",
                data=csv_qq,
                file_name=f"{file_prefix}_qq_plot_data.csv",
                mime="text/csv",
                key="download_qq_plot_data"
            )
        else:
            st.info("No Q-Q plot data available.")
    

    if export_param_coefficients:
        df_coef = None

        model_type = st.session_state.get("model_type")

        if model_type == "混合ワイブルモデル":
            df_coef = st.session_state.get("df_weibull_param_coefficients")

        elif model_type == "混合正規分布モデル":
            df_coef = st.session_state.get("df_param_coefficients")

        else:
            df_coef = st.session_state.get("df_param_coefficients")

        if df_coef is not None and not df_coef.empty:
            csv = df_coef.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download parameter coefficients CSV",
                data=csv,
                file_name=f"{base_filename}_parameter_coefficients.csv",
                mime="text/csv",
                key="download_parameter_coefficients_csv"
            )
        else:
            st.info("No parameter coefficient result to export.")
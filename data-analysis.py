import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from mixed_weibull import fit_by_throughput_percentile
from mixed_weibull import build_global_mixture_weibull_cdf
from mixed_weibull import make_weibull_param_coefficients
from mixed_weibull import plot_weibull_parameter_trends
from sampling import build_bivariate_sampler
from validation import make_qq_table_by_ttp, plot_qq_by_percentile_grid, plot_qq_by_ttp_grid
from quantile_correction import apply_quantile_correction

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
    # 既存キー
    "fleetdata_path": "/Users/hisashi/Desktop/data/data.xlsx",
    "throughput_quantile_path": "/Users/hisashi/Desktop/data/Percentile.xlsx",
    "fleetdata_sheet_names": [],
    "selected_feature": None,
    "df_fleetdata": None,
    "df_long": None,
    "df_throughput_quantile": None,

    # Tab3用
    "model_type": "混合ワイブルモデル",
    "selected_n_components": 2,
    "use_quantile_correction": False,
    "df_fit_results_dict": {},
    "global_cdf": None,
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

    # =========================
    # File path input
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

        if st.button("Check Fleet Data file"):
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

        if st.button("Check Total Throughput file"):
            try:
                path = Path(throughput_quantile_path)

                if not path.exists():
                    st.error("Throughput quantile file does not exist.")

                elif path.suffix.lower() not in [".xlsx", ".xls"]:
                    st.error("Please select an Excel file.")

                else:
                    st.success("Throughput quantile file checked successfully.")

            except Exception as e:
                st.error(f"Failed to check throughput quantile file: {e}")

    st.divider()

    # =========================
    # Feature sheet selection
    # =========================

    st.subheader("Feature selection")

    sheet_names = st.session_state["fleetdata_sheet_names"]

    if len(sheet_names) == 0:
        st.info("Fleet Data file を指定し、'Check fleet data file' を押してください。")

    else:
        selected_feature = st.selectbox(
            "解析したい特徴量",
            options=sheet_names,
            index=0
        )

        st.session_state["selected_feature"] = selected_feature

        st.write(f"選択中の解析対象: **{selected_feature}**")

        st.divider()

        # =========================
        # Load data
        # =========================

        if st.button("Load Data"):
            try:
                # df_fleetdata
                df_fleetdata = pd.read_excel(
                    st.session_state["fleetdata_path"],
                    sheet_name=selected_feature,
                    index_col=0
                )

                st.session_state["df_fleetdata"] = df_fleetdata

                # long format
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

                st.session_state["df_long"] = df_long

                # df_throughput_quantile
                df_throughput_quantile = pd.read_excel(
                    st.session_state["throughput_quantile_path"]
                )

                df_throughput_quantile.columns = [
                    "TTP_Percentile",
                    "TTP"
                ]

                st.session_state["df_throughput_quantile"] = df_throughput_quantile

                st.success("Data loaded successfully.")

            except Exception as e:
                st.error(f"Failed to load data: {e}")

    st.divider()

    # =========================
    # Data preview
    # =========================

    st.subheader("df_fleetdata")

    if st.session_state["df_fleetdata"] is not None:
        st.dataframe(
            st.session_state["df_fleetdata"],
            use_container_width=True
        )
    else:
        st.info("df_fleetdata is not loaded yet.")

    st.subheader("df_throughput_quantile")

    if st.session_state["df_throughput_quantile"] is not None:
        st.dataframe(
            st.session_state["df_throughput_quantile"],
            use_container_width=True
        )
    else:
        st.info("df_throughput_quantile is not loaded yet.")
        

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

        # =========================
        # Plot settings
        # =========================

        col1, col2 = st.columns([1, 3])

        with col1:

            st.subheader("Plot settings")

            st.markdown("#### Graph size")

            plot_height = st.slider(
                "Plot height",
                min_value=400,
                max_value=1000,
                value=700,
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
                color="TTP_Percentile",
                markers=True
            )

            fig.update_layout(
                height=plot_height,
                xaxis_title=selected_feature,
                yaxis_title="Cumulative Probability",
                legend_title="TTP_Percentile"
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

    df_long = st.session_state["df_long"]

    # =========================
    # Model selection
    # =========================

    st.subheader("Model selection")

    model_type = st.selectbox(
        "モデルを選択",
        options=[
            "混合ワイブルモデル",
            "混合正規分布モデル",
            "2次元スプライン補完",
        ],
        index=[
            "混合ワイブルモデル",
            "混合正規分布モデル",
            "2次元スプライン補完",
        ].index(st.session_state["model_type"])
        if st.session_state["model_type"] in [
            "混合ワイブルモデル",
            "混合正規分布モデル",
            "2次元スプライン補完",
        ]
        else 0
    )

    st.session_state["model_type"] = model_type

    st.divider()

    # =========================
    # Model-specific settings
    # =========================

    st.subheader("Model settings")

    if model_type == "混合ワイブルモデル":

        col1, col2 = st.columns([1, 1])

        with col1:
            current_n_components = st.session_state.get("selected_n_components", 2)

            if current_n_components not in [1, 2, 3]:
                current_n_components = 2

            selected_n_components = st.selectbox(
                "混合ワイブルの項数",
                options=[1, 2, 3],
                index=[1, 2, 3].index(current_n_components)
            )

            st.session_state["selected_n_components"] = selected_n_components

        with col2:
            weibull_param_fit_method = st.selectbox(
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
                        "weibull_param_fit_method",
                        "3次スプライン補完"
                    )
                )
            )

            st.session_state["weibull_param_fit_method"] = weibull_param_fit_method

        st.info(
            f"選択中: {model_type} / "
            f"{selected_n_components}項 / "
            f"TTP方向: {weibull_param_fit_method} / "
        )


    elif model_type == "2次元スプライン補完":

        st.info(
            "2次元スプライン補完では、次数と分位点補正の設定は使用しません。"
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

                    n_components = st.session_state["selected_n_components"]

                    df_fit = fit_by_throughput_percentile(
                        df_long,
                        n_components=n_components,
                    )

                    df_fit_results_dict = {
                        n_components: df_fit
                    }

                    st.session_state["df_fit_results_dict"] = df_fit_results_dict

                    if df_fit.empty:
                        st.session_state["global_cdf"] = None

                        st.warning(
                            f"混合ワイブルモデル {n_components}成分のフィッティング結果がありません。"
                        )

                    else:
                        method_ui = st.session_state["weibull_param_fit_method"]

                        # ==================================
                        # ワイブルパラメータ係数出力
                        # ==================================

                        method_map = {
                            "3次スプライン補完": "spline",
                            "3次多項式": "poly3"
                        }

                        param_fit_method = method_map[method_ui]

                        global_cdf = build_global_mixture_weibull_cdf(
                            df_fit_results_dict,
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

                        method_map = {
                            "3次スプライン補完": "spline",
                            "3次多項式": "poly3"
                        }

                        param_fit_method = method_map[
                            st.session_state["weibull_param_fit_method"]
                        ]

                        fig_param = plot_weibull_parameter_trends(
                            df_fit=df_fit,
                            n_components=n_components,
                            q_col="TTP_Percentile",
                            param_fit_method=method_map[st.session_state["weibull_param_fit_method"]]
                        )
                        st.session_state["df_weibull_param_coefficients"] = df_coef

                        st.success(
                            f"混合ワイブルモデル {n_components}成分のフィッティングとGlobal model作成が完了しました。"
                        )

                # =========================
                # 混合正規分布モデル
                # =========================
                elif model_type == "混合正規分布モデル":

                    n_components = st.session_state["selected_n_components"]

                    # TODO: 混合正規分布モデル関数を接続
                    # df_fit = fit_gaussian_mixture_by_throughput_percentile(
                    #     df_long,
                    #     n_components=n_components
                    # )
                    #
                    # df_fit_results_dict = {
                    #     n_components: df_fit
                    # }
                    #
                    # global_cdf = build_global_gaussian_mixture_cdf(
                    #     df_fit_results_dict,
                    #     n_components=n_components
                    # )

                    df_fit = pd.DataFrame()
                    df_fit_results_dict = {
                        n_components: df_fit
                    }

                    st.session_state["df_fit_results_dict"] = df_fit_results_dict
                    st.session_state["global_cdf"] = None

                    st.info("混合正規分布モデルはまだ未実装です。")

                # =========================
                # 2次元スプライン補完
                # =========================
                elif model_type == "2次元スプライン補完":

                    # TODO: 2次元スプライン補完関数を接続
                    # spline_model = fit_2d_spline_model(df_long)

                    spline_model = None

                    st.session_state["global_cdf"] = spline_model
                    st.session_state["df_fit_results_dict"] = {}

                    st.info("2次元スプライン補完モデルはまだ未実装です。")

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
            with st.expander(f"{n} component fit result"):
                if df_fit is not None and not df_fit.empty:
                    st.dataframe(df_fit, use_container_width=True)
                else:
                    st.info("No fit result.")
    else:
        st.info("No fitting result yet.")

    
    st.subheader("Weibull parameter coefficients")

    df_coef = st.session_state.get("df_weibull_param_coefficients")

    if df_coef is not None and not df_coef.empty:

        st.dataframe(
            df_coef,
            use_container_width=True
        )

    else:
        st.info("No Weibull parameter coefficient result yet.")


    st.subheader("Weibull parameter trends")
    if st.session_state["df_fit_results_dict"]:

        n_components = st.session_state["selected_n_components"]

        df_fit = st.session_state["df_fit_results_dict"].get(n_components)

        if df_fit is not None and not df_fit.empty:

            method_map = {
                "3次スプライン補完": "spline",
                "3次多項式": "poly3"
            }

            fig_param = plot_weibull_parameter_trends(
                df_fit=df_fit,
                n_components=n_components,
                q_col="TTP_Percentile",
                param_fit_method=method_map[st.session_state["weibull_param_fit_method"]]
            )

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

    selected_feature = st.session_state.get("selected_feature", "value")

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

    st.divider()

    # =========================
    # Pre-check
    # =========================

    if st.session_state["global_cdf"] is None:
        st.info("Please run fitting first.")

    elif st.session_state["df_throughput_quantile"] is None:
        st.info("Please load Total Throughput quantile data first.")

    else:
        if st.button("Generate random samples", key="generate_random_samples"):
            try:
                sampler = build_bivariate_sampler(
                    df_throughput_quantile=st.session_state["df_throughput_quantile"],
                    global_cdf=st.session_state["global_cdf"],
                    q_col="TTP_Percentile",
                    throughput_col="TTP"
                )

                df_sample = sampler(
                    n_samples=st.session_state["n_samples"],
                    random_state=st.session_state["random_state"]
                )

                # =========================
                # Quantile correction
                # =========================

                if st.session_state["use_quantile_correction"]:
                    df_sample = apply_quantile_correction(
                        df_sample=df_sample,
                        df_reference=st.session_state["df_long"],
                        global_cdf=st.session_state["global_cdf"],
                        q_col="TTP_Percentile",
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
        
        # =========================
        # Visualization
        # =========================

        if st.session_state["df_sample"] is not None:
            df_sample = st.session_state["df_sample"]

            if not df_sample.empty:

                st.subheader("Generated sample visualization")

                value_min = float(df_sample["value"].min())
                value_max = float(df_sample["value"].max())
                ttp_min = float(df_sample["TTP"].min())
                ttp_max = float(df_sample["TTP"].max())

                # =========================
                # Histogram
                # =========================

                st.markdown("### Histogram")

                col_hist_setting, col_hist_plot = st.columns([1, 3])

                with col_hist_setting:
                    # st.markdown("#### Histogram settings")

                    hist_x_min = st.number_input(
                        "Hist X min",
                        value=value_min,
                        key="hist_x_min"
                    )

                    hist_x_max = st.number_input(
                        "Hist X max",
                        value=value_max,
                        key="hist_x_max"
                    )

                    hist_y_min = st.number_input(
                        "Hist Y min",
                        value=0.0,
                        key="hist_y_min"
                    )

                    hist_y_max = st.number_input(
                        "Hist Y max",
                        value=float(len(df_sample)),
                        key="hist_y_max"
                    )

                    hist_bin_width = st.number_input(
                        "Bin width",
                        min_value=1e-12,
                        value=(value_max - value_min) / 50,
                        key="hist_bin_width"
                    )

                with col_hist_plot:
                    fig_hist = px.histogram(
                        df_sample,
                        x="value",
                        nbins=None
                    )

                    fig_hist.update_traces(
                        xbins=dict(
                            start=hist_x_min,
                            end=hist_x_max,
                            size=hist_bin_width
                        )
                    )

                    fig_hist.update_layout(
                        height=500,
                        xaxis_title=selected_feature,
                        yaxis_title="Count",
                        showlegend=False
                    )

                    fig_hist.update_xaxes(
                        range=[hist_x_min, hist_x_max]
                    )

                    fig_hist.update_yaxes(
                        range=[hist_y_min, hist_y_max]
                    )

                    st.plotly_chart(
                        fig_hist,
                        use_container_width=True
                    )

                # =========================
                # Scatter
                # =========================

                st.markdown("### Scatter plot")

                col_scatter_setting, col_scatter_plot = st.columns([1, 3])

                with col_scatter_setting:
                    # st.markdown("#### Scatter settings")

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

    df_long = st.session_state["df_long"]
    df_sample = st.session_state["df_sample"]
    selected_feature = st.session_state.get("selected_feature", "value")

    if df_long is None:
        st.info("Please load data first.")

    elif df_sample is None or df_sample.empty:
        st.info("Please generate random samples first.")

    else:
        st.subheader("Validation settings")

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
                df_qq = make_qq_table_by_ttp(
                    df_long=df_long,
                    df_sample=df_sample,
                    percentiles=target_percentiles,
                    q_col="TTP_Percentile",
                    p_col="value_Percentile",
                    value_col="value",
                    ttp_bin_width=2.5
                )

                st.session_state["df_validation"] = df_qq

                if df_qq.empty:
                    st.warning("Q-Q table is empty. Check TTP_Percentile and value_Percentile values.")
                else:
                    st.success("Validation completed.")

            except Exception as e:
                st.error(f"Validation failed: {e}")

        st.divider()

        # =========================
        # Q-Q Plot
        # =========================

        st.subheader("Q-Q plot")

        df_qq = st.session_state.get("df_validation")

        if df_qq is not None and not df_qq.empty:
            try:
                fig_qq = plot_qq_by_ttp_grid(
                    df_qq=df_qq,
                    selected_feature=selected_feature,
                    q_col="TTP_Percentile",
                    p_col="value_Percentile",
                    n_cols=4
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
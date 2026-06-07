import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from mixed_weibull import fit_by_throughput_percentile
from mixed_weibull import build_global_mixture_weibull_cdf
from sampling import build_bivariate_sampler
from validation import make_validation_table, make_validation_metrics, plot_validation_diagnostics

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
    "fleetdata_path": "/Users/hisashi/python_projects/data_science/アプリ/data/fleetdata.xlsx",
    "throughput_quantile_path": "/Users/hisashi/python_projects/data_science/アプリ/data/TTP_Percentile.xlsx",
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

    if model_type in ["混合ワイブルモデル", "混合正規分布モデル"]:

        col1, col2 = st.columns([1, 1])

        with col1:
            current_n_components = st.session_state.get("selected_n_components", 2)

            if current_n_components not in [1, 2, 3]:
                current_n_components = 2

            selected_n_components = st.selectbox(
                "次数",
                options=[1, 2, 3],
                index=[1, 2, 3].index(current_n_components)
            )

            st.session_state["selected_n_components"] = selected_n_components

        with col2:
            use_quantile_correction = st.checkbox(
                "分位点補正を行う",
                value=st.session_state["use_quantile_correction"]
            )

            st.session_state["use_quantile_correction"] = use_quantile_correction

        st.info(
            f"選択中: {model_type} / {selected_n_components}成分 / "
            f"分位点補正: {'あり' if use_quantile_correction else 'なし'}"
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
                        global_cdf = build_global_mixture_weibull_cdf(
                            df_fit_results_dict,
                            n_components=n_components
                        )

                        st.session_state["global_cdf"] = global_cdf

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
        hist_bins = st.number_input(
            "Histogram bins",
            min_value=5,
            max_value=500,
            value=st.session_state["hist_bins"],
            step=5
        )

    st.session_state["n_samples"] = n_samples
    st.session_state["random_state"] = random_state
    st.session_state["hist_bins"] = hist_bins

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

                st.session_state["df_sample"] = df_sample

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
                    st.markdown("#### Histogram settings")

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
                        height=400,
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
                    st.markdown("#### Scatter settings")

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
                        height=550,
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

        # # =========================
        # # Visualization
        # # =========================

        # if st.session_state["df_sample"] is not None:
        #     df_sample = st.session_state["df_sample"]

        #     if not df_sample.empty:

        #         st.subheader("Generated sample visualization")

        #         st.subheader("Histogram")

        #         fig_hist = px.histogram(
        #             df_sample,
        #             x="value",
        #             nbins=st.session_state["hist_bins"]
        #         )

        #         fig_hist.update_layout(
        #             height=400,
        #             xaxis_title=selected_feature,
        #             yaxis_title="Count",
        #             showlegend=False,
        #         )

        #         st.plotly_chart(
        #             fig_hist,
        #             use_container_width=True
        #         )

        #         st.subheader("Scatter plot")

        #         fig_scatter = px.scatter(
        #             df_sample,
        #             x="TTP",
        #             y="value",
        #             opacity=0.35
        #         )

        #         fig_scatter.update_layout(
        #             height=500,
        #             xaxis_title="TTP",
        #             yaxis_title=selected_feature,
        #             showlegend=False
        #         )

        #         st.plotly_chart(
        #             fig_scatter,
        #             use_container_width=True
        #         )

        #         st.subheader("Generated sample summary")

        #         st.dataframe(
        #             df_sample.describe(),
        #             use_container_width=True
        #         )



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

        col1, col2 = st.columns([1, 1])

        with col1:
            n_bins = st.number_input(
                "TTP bin count",
                min_value=3,
                max_value=30,
                value=8,
                step=1
            )

        with col2:
            st.write("")
            st.write("")
            run_validation = st.button(
                "Run validation",
                key="run_validation"
            )

        if run_validation:
            try:
                df_validation = make_validation_table(
                    df_long=df_long,
                    df_sample=df_sample,
                    q_col="TTP_Percentile",
                    p_col="value_Percentile",
                    value_col="value",
                    n_bins=n_bins
                )

                df_metrics = make_validation_metrics(
                    df_validation
                )

                st.session_state["df_validation"] = df_validation
                st.session_state["df_validation_metrics"] = df_metrics

                st.success("Validation completed.")

            except Exception as e:
                st.error(f"Validation failed: {e}")

        st.divider()

        if st.session_state.get("df_validation") is not None:

            df_validation = st.session_state["df_validation"]
            df_metrics = st.session_state["df_validation_metrics"]

            st.subheader("Validation metrics")

            if df_metrics is not None and not df_metrics.empty:
                st.dataframe(
                    df_metrics,
                    use_container_width=True
                )
            else:
                st.info("No metrics available.")

            st.subheader("Validation plots")

            try:
                fig_val = plot_validation_diagnostics(
                    df_long=df_long,
                    df_sample=df_sample,
                    df_validation=df_validation,
                    selected_feature=selected_feature,
                    q_col="TTP_Percentile",
                    p_col="value_Percentile",
                    value_col="value"
                )

                st.plotly_chart(
                    fig_val,
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Failed to plot validation diagnostics: {e}")

            st.subheader("Validation table")

            if df_validation is not None and not df_validation.empty:
                st.dataframe(
                    df_validation,
                    use_container_width=True
                )
            else:
                st.info("No validation result.")


# =========================
# Tab 6: Export
# =========================

with tab_export:
    st.header("6. Export")

    st.subheader("Download results")

    if st.session_state["df_long"] is not None:
        csv_long = st.session_state["df_long"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download long format data",
            data=csv_long,
            file_name="dod_long_data.csv",
            mime="text/csv"
        )

    if (
        st.session_state["df_sample"] is not None
        and not st.session_state["df_sample"].empty
    ):
        csv_sample = st.session_state["df_sample"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download generated samples",
            data=csv_sample,
            file_name="generated_samples.csv",
            mime="text/csv"
        )

    if (
        st.session_state["df_corrected"] is not None
        and not st.session_state["df_corrected"].empty
    ):
        csv_corrected = st.session_state["df_corrected"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download corrected samples",
            data=csv_corrected,
            file_name="corrected_samples.csv",
            mime="text/csv"
        )
import streamlit as st
import altair as alt

import functions.data_processor as data_processor
import functions.visualisation as visualisation

from streamlit_extras.dataframe_explorer import dataframe_explorer
import utils.design_format as format
import utils.utils as utils
import datetime


def run_theme_tab(uploaded_data_filtered):

    st.write()

    visualisation.show_theme_metrics(uploaded_data_filtered)

    format.horizontal_line()

    tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Count Analysis", "Mean Analysis", "Sum Analysis"])

    with tab1:
        st.altair_chart(
            visualisation.plot_theme_heatmap(uploaded_data_filtered),
            use_container_width=True,
        )

    with tab2:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Count Analysis")

        st.markdown("###### Count of Articles by Theme")
        theme_count_chart = visualisation.plot_theme_timeseries(
            uploaded_data_filtered, "count()", "Number of Articles"
        )
        st.altair_chart(theme_count_chart, use_container_width=True)

        format.horizontal_line()

    with tab3:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Mean Analysis")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Mean of Facebook Interactions by Theme")
            theme_mean_chart = visualisation.plot_theme_timeseries(
                uploaded_data_filtered,
                "mean(facebook_interactions)",
                "Mean of Facebook Interactions",
            )
            st.altair_chart(theme_mean_chart, use_container_width=True)

        with col2:
            df_mean = (
                uploaded_data_filtered.groupby(["theme", "date_extracted"])[
                    "facebook_interactions"
                ]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_mean"):
                col1, col2 = st.columns([2, 1])
                threshold = st.slider(
                    "Mean number of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_mean.facebook_interactions.max()),
                    value=1000,
                    step=100,
                )
                st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_theme(
                        df_mean,
                        uploaded_data_filtered,
                        "facebook_interactions",
                        threshold,
                    ),
                )

        # altair chart for pct change in mean of interactions
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Mean of Facebook Interactions by Theme"
            )
            df_mean_agg = data_processor.aggregate_pct_change(
                uploaded_data_filtered,
                ["theme", "date_extracted"],
                "facebook_interactions",
                "mean",
            )
            theme_pct_change_chart = visualisation.plot_theme_timeseries(
                df_mean_agg,
                "pct_change",
                "Percent Change in Mean of Facebook Interactions %",
            )
            st.altair_chart(theme_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_mean_pct_change"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_mean_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_theme(
                        df_mean_agg, uploaded_data_filtered, "pct_change", threshold
                    ),
                )

        format.horizontal_line()

    with tab4:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Sum Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Sum of Facebook Interactions by Theme")
            theme_sum_chart = visualisation.plot_theme_timeseries(
                uploaded_data_filtered,
                "sum(facebook_interactions)",
                "Sum of Facebook Interactions",
            )
            st.altair_chart(theme_sum_chart, use_container_width=True)
        with col2:
            df_sum = (
                uploaded_data_filtered.groupby(["theme", "date_extracted"])[
                    "facebook_interactions"
                ]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_sum"):
                threshold = st.slider(
                    "Sum of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_sum.facebook_interactions.max()),
                    value=1000,
                    step=100,
                )
                submit_threshold_sum = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_theme(
                        df_sum,
                        uploaded_data_filtered,
                        "facebook_interactions",
                        threshold,
                    ),
                )

        # altair chart for pct change in sum of interactions
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Sum of Facebook Interactions by Theme"
            )
            df_sum_agg = data_processor.aggregate_pct_change(
                uploaded_data_filtered,
                ["theme", "date_extracted"],
                "facebook_interactions",
                "sum",
            )
            theme_pct_change_chart = visualisation.plot_theme_timeseries(
                df_sum_agg,
                "pct_change",
                "Percent Change in Sum of Facebook Interactions %",
            )
            st.altair_chart(theme_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_sum_pct_change"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_sum_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold_sum_pct = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_theme(
                        df_sum_agg, uploaded_data_filtered, "pct_change", threshold
                    ),
                )
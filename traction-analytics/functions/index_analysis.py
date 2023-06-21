import streamlit as st
import altair as alt

import functions.data_processor as data_processor
import functions.visualisation as visualisation

from streamlit_extras.dataframe_explorer import dataframe_explorer
import utils.design_format as format
import utils.utils as utils
import datetime

def run_index_tab(uploaded_data_filtered):
    # get list of themes sorted by sum of interactions
    themes_sorted = uploaded_data_filtered.groupby("theme")[
        "facebook_interactions"
    ].sum()
    themes_sorted = themes_sorted.sort_values(ascending=False)
    themes_sorted = themes_sorted.index.tolist()

    selected_theme = st.selectbox("Select Theme", options=themes_sorted)

    theme_data = data_processor.filter_data_by_theme(
        uploaded_data_filtered, selected_theme
    )

    visualisation.show_index_metrics(theme_data)

    format.horizontal_line()

    tab1, tab2, tab3, tab4= st.tabs(["Heatmap", "Count Analysis", "Mean Analysis", "Sum Analysis"])

    with tab1:
        st.altair_chart(
            visualisation.plot_index_heatmap(theme_data),
            use_container_width=True,
        )

    with tab2:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Count Analysis")

        st.markdown("###### Count of Articles by Index")
        index_count_chart = visualisation.plot_index_timeseries(
            theme_data, "count()", "Number of Articles"
        )

        st.altair_chart(index_count_chart, use_container_width=True)

        format.horizontal_line()

    with tab3:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Mean Analysis")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Mean of Facebook Interactions by Index")
            index_mean_chart = visualisation.plot_index_timeseries(
                theme_data,
                "mean(facebook_interactions)",
                "Mean of Facebook Interactions",
            )

            st.altair_chart(index_mean_chart, use_container_width=True)
        with col2:
            df_mean = (
                theme_data.groupby(["index", "date_extracted"])["facebook_interactions"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_mean_index"):
                threshold = st.slider(
                    "Mean of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_mean["facebook_interactions"].max()),
                    value=100,
                    step=10,
                )
                submit_threshold_mean = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_index(
                        df_mean, theme_data, "facebook_interactions", threshold
                    ),
                )

        # pct change of mean

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Mean of Facebook Interactions by Index"
            )
            df_mean_agg = data_processor.aggregate_pct_change(
                theme_data, ["index", "date_extracted"], "facebook_interactions", "mean"
            )
            index_pct_change_chart = visualisation.plot_index_timeseries(
                df_mean_agg,
                "pct_change",
                "Percent Change in Mean of Facebook Interactions %",
            )
            st.altair_chart(index_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_pct_change_index"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_mean_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold_pct_change = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_index(
                        df_mean_agg, theme_data, "pct_change", threshold
                    ),
                )

        format.horizontal_line()

    with tab4:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Sum Analysis")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Sum of Facebook Interactions by Index")
            index_sum_chart = visualisation.plot_index_timeseries(
                theme_data, "sum(facebook_interactions)", "Sum of Facebook Interactions"
            )

            st.altair_chart(index_sum_chart, use_container_width=True)
        with col2:
            df_sum = (
                theme_data.groupby(["index", "date_extracted"])["facebook_interactions"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_sum_index"):
                threshold = st.slider(
                    "Sum of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_sum.facebook_interactions.max()),
                    value=1000,
                    step=100,
                )
                submit_threshold_sum = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_index(
                        df_sum, theme_data, "facebook_interactions", threshold
                    ),
                )

        # pct change of sum

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Sum of Facebook Interactions by Index"
            )
            df_sum_agg = data_processor.aggregate_pct_change(
                theme_data, ["index", "date_extracted"], "facebook_interactions", "sum"
            )
            index_pct_change_chart = visualisation.plot_index_timeseries(
                df_sum_agg,
                "pct_change",
                "Percent Change in Sum of Facebook Interactions %",
            )
            st.altair_chart(index_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_pct_change_sum_index"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_sum_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold_pct_change = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=visualisation.show_articles_exceeding_threshold_index(
                        df_sum_agg, theme_data, "pct_change", threshold
                    ),
                )
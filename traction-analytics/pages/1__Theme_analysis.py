import streamlit as st
import altair as alt

import functions.data_processor as data_processor
import functions.visualisation as visualisation

from streamlit_extras.dataframe_explorer import dataframe_explorer
import utils.design_format as format
import utils.utils as utils
import datetime

st.set_page_config(page_title="Theme and Index Analysis", page_icon="📰", layout="wide")

st.title("🖥️ Theme and Index Analysis")
format.horizontal_line()
format.align_text(
    """
    In this page, you can view the time series data analysis of the themes and index.
    """,
    "justify",
)

format.horizontal_line()


def run_theme_tab(uploaded_data_filtered):
    st.write()

    visualisation.show_theme_metrics(uploaded_data_filtered)

    format.horizontal_line()

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        st.subheader("Count Analysis")

    st.markdown("###### Count of Articles by Theme")
    theme_count_chart = visualisation.plot_theme_timeseries(
        uploaded_data_filtered, "count()", "Number of Articles"
    )
    st.altair_chart(theme_count_chart, use_container_width=True)

    format.horizontal_line()

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        st.subheader("Mean Analysis")

    st.markdown("###### Mean of Facebook Interactions by Theme")
    theme_mean_chart = visualisation.plot_theme_timeseries(
        uploaded_data_filtered, "mean(facebook_interactions)", "Mean of Facebook Interactions"
    )
    st.altair_chart(theme_mean_chart, use_container_width=True)

    # altair chart for pct change in mean of interactions
    st.markdown("###### Percent Change in Mean of Facebook Interactions by Theme")
    df_mean_agg = data_processor.aggregate_pct_change(uploaded_data_filtered, ["theme", "date_extracted"], "facebook_interactions", "mean")
    theme_pct_change_chart = visualisation.plot_theme_timeseries(
        df_mean_agg, "pct_change", "Percent Change in Mean of Facebook Interactions"
    )
    st.altair_chart(theme_pct_change_chart, use_container_width=True)

    format.horizontal_line()

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        st.subheader("Sum Analysis")

    st.markdown("###### Sum of Facebook Interactions by Theme")
    theme_sum_chart = visualisation.plot_theme_timeseries(
        uploaded_data_filtered, "sum(facebook_interactions)", "Sum of Facebook Interactions"
    )
    st.altair_chart(theme_sum_chart, use_container_width=True)

    # altair chart for pct change in sum of interactions
    st.markdown("###### Percent Change in Sum of Facebook Interactions by Theme")
    df_sum_agg = data_processor.aggregate_pct_change(uploaded_data_filtered, ["theme", "date_extracted"], "facebook_interactions", "sum")
    theme_pct_change_chart = visualisation.plot_theme_timeseries(
        df_sum_agg, "pct_change", "Percent Change in Sum of Facebook Interactions"
    )
    st.altair_chart(theme_pct_change_chart, use_container_width=True)





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
    
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        st.subheader("Count Analysis")

    st.markdown("###### Count of Articles by Index")
    index_count_chart = visualisation.plot_index_timeseries(
        theme_data, "count()", "Number of Articles"
    )

    st.altair_chart(index_count_chart, use_container_width=True)

    format.horizontal_line()

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        st.subheader("Mean Analysis")

    st.markdown("###### Mean of Facebook Interactions by Index")
    index_mean_chart = visualisation.plot_index_timeseries(
        theme_data, "mean(facebook_interactions)", "Mean of Facebook Interactions"
    )

    st.altair_chart(index_mean_chart, use_container_width=True)

    # pct change of mean

    st.markdown("###### Percent Change in Mean of Facebook Interactions by Index")
    df_mean_agg = data_processor.aggregate_pct_change(theme_data, ["index", "date_extracted"], "facebook_interactions", "mean")
    index_pct_change_chart = visualisation.plot_index_timeseries(
        df_mean_agg, "pct_change", "Percent Change in Mean of Facebook Interactions"
    )
    st.altair_chart(index_pct_change_chart, use_container_width=True)

    format.horizontal_line()
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        st.subheader("Sum Analysis")

    st.markdown("###### Sum of Facebook Interactions by Index")
    index_sum_chart = visualisation.plot_index_timeseries(
        theme_data, "sum(facebook_interactions)", "Sum of Facebook Interactions"
    )

    st.altair_chart(index_sum_chart, use_container_width=True)

    # pct change of sum

    st.markdown("###### Percent Change in Sum of Facebook Interactions by Index")
    df_sum_agg = data_processor.aggregate_pct_change(theme_data, ["index", "date_extracted"], "facebook_interactions", "sum")
    index_pct_change_chart = visualisation.plot_index_timeseries(
        df_sum_agg, "pct_change", "Percent Change in Sum of Facebook Interactions"
    )
    st.altair_chart(index_pct_change_chart, use_container_width=True)



def run():
    if utils.check_session_state_key("csv_file"):
        uploaded_data = utils.get_cached_object("csv_file")
        uploaded_data = data_processor.process_data(uploaded_data)
        with st.expander("View Raw Data"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(label="Number of Rows", value=uploaded_data.shape[0])
            st.dataframe(dataframe_explorer(uploaded_data), use_container_width=True)

        max_date = uploaded_data["date"].max()
        min_date = uploaded_data["date"].min()
        max_interactions = uploaded_data["facebook_interactions"].max()
        min_interactions = uploaded_data["facebook_interactions"].min()

        with st.sidebar:
            with st.form("min_interactions"):
                st.subheader("Filters")
                st.write("Input filters for data analysis.")

                min_interactions = st.slider(
                    "Minimum Number of Facebook Interactions per article",
                    min_value=0,
                    max_value=10000,
                    value=1000,
                )
                date_range = st.date_input(
                    "Date range of articles",
                    value=(
                        min_date,
                        max_date,
                    ),
                    min_value=min_date,
                    max_value=max_date,
                )

                selected_themes = st.multiselect(
                    "Select Themes to Exclude",
                    options=uploaded_data["theme"].unique().tolist(),
                    default=["general"],
                )

                submit_button = st.form_submit_button(label="Submit")

        uploaded_data_filtered = data_processor.filter_data(
            uploaded_data, min_interactions, date_range, selected_themes
        )

        tab1, tab2 = st.tabs(["Theme Analysis", "Index Analysis"])

        with tab1:
            run_theme_tab(uploaded_data_filtered)

        with tab2:
            run_index_tab(uploaded_data_filtered)

    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    with st.spinner("Loading Dashboard..."):
        run()

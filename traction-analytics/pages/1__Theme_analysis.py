import streamlit as st
import altair as alt

import functions.data_processor as data_processor
import functions.visualisation as visualisation
import functions.theme_analysis as theme_analysis
import functions.index_analysis as index_analysis

from streamlit_extras.dataframe_explorer import dataframe_explorer
import utils.design_format as format
import utils.utils as utils
import datetime

st.set_page_config(
    page_title="Theme and Index Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🖥️ Theme and Index Analysis")
format.horizontal_line()
format.align_text(
    """
    In this page, you can view the time series data analysis of the themes and index.
    """,
    "justify",
)

format.horizontal_line()

def run_summary_tab(uploaded_data_filtered):
    st.write()

    visualisation.show_summary_metrics(uploaded_data_filtered)

    format.horizontal_line()

    st.altair_chart(
        visualisation.plot_heatmap(uploaded_data_filtered), use_container_width=True
    )

    return

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
                    value=500,
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
                    options=sorted(uploaded_data["theme"].unique().tolist()),
                    default=["general"],
                )

                selected_index = st.multiselect(
                    "Select Index to Exclude",
                    options=sorted(uploaded_data["index"].unique().tolist()),
                    default=["others"],
                )

                submit_button = st.form_submit_button(label="Submit")

        uploaded_data_filtered = data_processor.filter_data(
            uploaded_data, min_interactions, date_range, selected_themes, selected_index
        )



        tab1, tab2, tab3 = st.tabs(["Summary Analysis", "Theme Analysis", "Index Analysis"])

        with tab1:
            run_summary_tab(uploaded_data_filtered)

        with tab2:
            theme_analysis.run_theme_tab(uploaded_data_filtered)

        with tab3:
            index_analysis.run_index_tab(uploaded_data_filtered)

    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    # with st.spinner("Loading Dashboard..."):
    run()

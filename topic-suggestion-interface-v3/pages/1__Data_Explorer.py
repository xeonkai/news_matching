import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
import utils.design_format as format
import utils.utils as utils

st.set_page_config(page_title="Data Explorer", page_icon="📰", layout="wide")

st.title("🔎 Data Explorer")
format.horizontal_line()
format.align_text(
    """
    In this page, you are able to explore the uploaded CSV DataFrame.
    """,
    "justify",
)

format.horizontal_line()


def run():
    if utils.check_session_state_key("csv_file"):
        uploaded_data = utils.get_cached_object("csv_file")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(label="Number of Rows", value=uploaded_data.shape[0])
        with col2:
            st.metric(label="Number of Columns", value=uploaded_data.shape[1])
        st.dataframe(dataframe_explorer(uploaded_data), use_container_width=True)
    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    run()

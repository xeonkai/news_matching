import streamlit as st
import utils.design_format as format
import utils.utils as utils

st.set_page_config(
    page_title="Traction Analytics Interface Demo", page_icon="📰", layout="wide"
)

st.title("🖥️ Traction Analytics Interface Demo")
format.horizontal_line()
st.subheader("Welcome!")
format.align_text(
    """
    This is a demo of the Traction Analytics Interface. Upload a CSV file of the news daily scans below to begin.
    """,
    "justify",
)

format.horizontal_line()

st.write("")


def run():
    utils.csv_file_uploader()

    st.write("")

    format.horizontal_line()


if __name__ == "__main__":
    run()

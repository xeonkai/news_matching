import os
import streamlit as st
from st_pages import add_page_title
from dotenv import load_dotenv

load_dotenv()

GSHEET_TAXONOMY_ID = os.environ.get("GSHEET_TAXONOMY_ID")
gsheet_taxonomy_url = "https://docs.google.com/spreadsheets/d/" + GSHEET_TAXONOMY_ID


add_page_title(layout="wide")

st.markdown("""---""")
st.warning(f"Taxonomy explorer moved to [google sheets]({gsheet_taxonomy_url})")
st.markdown("""---""")

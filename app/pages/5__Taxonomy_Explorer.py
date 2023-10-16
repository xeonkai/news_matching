import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox
from streamlit_extras.dataframe_explorer import dataframe_explorer
from utils.core import (
    fetch_taxonomy,
    save_taxonomy,
    fetch_latest_taxonomy,
    list_taxonomies,
)
from st_pages import add_page_title

add_page_title(layout="wide")

# st.set_page_config(page_title="Taxonomy Explorer", page_icon="📰", layout="wide")

# st.title("🔮 Taxonomy Explorer")
st.markdown("""---""")
st.markdown(
    """
    In this page, you can explore the taxonomy containing the Theme and Index from the range of data selected. You may filter the table by Theme and Index.
    """
)
st.markdown("""---""")


def run():
    st.subheader("Previewing the Taxonomy")

    taxonomy_date_sorted = list_taxonomies()

    past_taxonomies_selection = selectbox(
        "Taxonomy versions", taxonomy_date_sorted, no_selection_label="<Latest>"
    )

    if past_taxonomies_selection is None:
        taxonomy_df = fetch_latest_taxonomy()
    else:
        taxonomy_df = fetch_taxonomy(past_taxonomies_selection)

    taxonomy_chains_df = dataframe_explorer(taxonomy_df)

    edit_df = st.data_editor(
        taxonomy_chains_df, num_rows="dynamic", use_container_width=True
    )

    save_btn = st.button("Save taxonomy")
    if save_btn:
        save_taxonomy(edit_df)


if __name__ == "__main__":
    run()

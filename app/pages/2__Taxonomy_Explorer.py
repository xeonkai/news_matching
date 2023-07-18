import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
from utils.core import generate_taxonomy

st.set_page_config(page_title="Taxonomy Explorer", page_icon="📰", layout="wide")

st.title("🔮 Taxonomy Explorer")
st.markdown("""---""")
st.markdown(
    """
    In this page, you can explore the taxonomy containing the Theme and Index from the range of data selected. You may filter the table by Theme and Index.
    """
)
st.markdown("""---""")


def run():
    st.subheader("Previewing the Taxonomy")

    taxonomy_chains_df = generate_taxonomy()

    taxonomy_chains_df_explorer = dataframe_explorer(taxonomy_chains_df)

    st.dataframe(taxonomy_chains_df_explorer, use_container_width=True)


if __name__ == "__main__":
    run()

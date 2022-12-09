import streamlit as st
import pandas as pd

st.title("Top 10 Articles")

df_filtered = st.session_state["df_filtered"]

top_articles = df_filtered.sort_values(by = ["Facebook Interactions"], ascending = False).head(10)

st.dataframe(top_articles[["Published", "Headline", "Summary", "Link", "Domain", "Facebook Interactions"]])
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title = "Topic Modelling",
                  layout="wide")


##### main page #####
selected_model = st.selectbox("Choose a model:", ["Top2Vec"])

title_or_content = st.selectbox(
    "Compare Title or Content",
    (
        "Title",
        "Content"
    ),
)

st.button("Load model!")


selected_model_path = Path("data", "intermediate_data", "selected_model_variable.pickle")
title_or_content_path = Path("data", "intermediate_data", "title_or_content_variable.pickle")

with open(selected_model_path, 'wb') as file:
    pickle.dump(selected_model, file)

with open(title_or_content_path, 'wb') as file:
    pickle.dump(title_or_content, file)

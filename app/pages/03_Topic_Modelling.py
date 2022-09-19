import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

st.set_page_config(page_title = "Topic Modelling",
                  layout="wide")


##### main page #####
selected_model = st.selectbox("Choose a model:", ["Top2Vec"])

st.button("Load model!")
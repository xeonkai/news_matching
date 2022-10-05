import streamlit as st

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

final_button = st.button("Load model!")

st.session_state["title_or_content"] = title_or_content
st.session_state["selected_model"] = selected_model

if final_button:
        "Proceed with Topic Discovery!"

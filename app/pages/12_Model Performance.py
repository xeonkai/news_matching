import streamlit as st
import pandas as pd
import numpy as np
from utils import core
import altair as alt


def plot_top_k_accuracy(top_k=20):
    filerhandler = core.FileHandler(core.DATA_DIR)

    if len(filerhandler) == 0:
        return

    labelled_df = filerhandler.query(
        "SELECT headline,link,label FROM daily_news WHERE label is not NULL"
    )

    if len(labelled_df) == 0:
        return

    classification_model = core.load_classification_model()
    df = labelled_df.pipe(core.label_df, classification_model, column="headline")
    k_range = list(range(1, top_k))

    top_k_scores = [
        np.mean(
            [
                label in pred
                for label, pred in zip(df["label"], df["suggested_labels"].str[:k])
            ]
        )
        for k in k_range
    ]
    top_k_df = pd.DataFrame({"k": k_range, "accuracy": top_k_scores})
    chart = (
        alt.Chart(top_k_df)
        .mark_line(point=True)
        .encode(x="k:Q", y="accuracy:Q")
        .properties(title="Top k-accuracy of predicted vs past labelled")
    )
    st.altair_chart(chart)
    return chart


if __name__ == "__main__":
    k = st.sidebar.number_input("Top k accuracy to check", value=20)
    plot_top_k_accuracy(k)

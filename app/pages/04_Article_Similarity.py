import time

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

st.set_page_config(page_title="Article Similarity", layout="wide")

st.title("Article Similarity")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()


class SearchJaccard:
    """Jaccard similarity based on tokenized word sets"""

    def __init__(
        self,
        checkpoint,
        corpus: list[str],
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # Save corpus for finding original text
        self.df = pd.DataFrame(corpus, columns=["corpus"])
        self.df["token_set"] = self.df["corpus"].apply(
            lambda doc: set(self.tokenizer.tokenize(doc))
        )

    def __call__(self, text: str, k: int):
        text_token_set = set(self.tokenizer.tokenize(text))
        similarity = self.df["token_set"].apply(
            self.jaccard_similarity, B=text_token_set
        )
        top_k = similarity.nlargest(k)
        indexes = top_k.index.values
        distances = top_k.values

        return distances, indexes

    def __getitem__(self, ids):
        return self.df.loc[ids, "corpus"].to_list()

    # Reduce memory usage by storing corpus only in df, extract when needed
    @property
    def corpus(self):
        return self.df["corpus"].to_list()

    @staticmethod
    def jaccard_similarity(A, B):
        # Find intersection of two sets
        nominator = A.intersection(B)

        # Find union of two sets
        denominator = A.union(B)

        # Take the ratio of sizes
        similarity = len(nominator) / len(denominator)

        return similarity


class SearchFlatL2:
    """Exhaustive euclidean search on flat vector index"""

    def __init__(
        self,
        checkpoint,
        sentence_embeddings,
    ):
        self.model = SentenceTransformer(checkpoint)

        d = sentence_embeddings.shape[1]

        # IndexFlatL2 config
        self.index = faiss.IndexFlatL2(d)

        # faiss indexing embeddings
        self.index.add(sentence_embeddings)

    def __call__(self, text: str, k: int):

        xq = self.model.encode([text])
        D, I = self.index.search(xq, k)

        return D[0], I[0]


df = st.session_state["df_filtered"]
df["Published"] = pd.to_datetime(df["Published"])

# # Set model
checkpoint = "multi-qa-MiniLM-L6-cos-v1"


@st.cache(allow_output_mutation=True)
def load_search_methods(search_type, checkpoint, df):
    doc = df["Headline"].to_list()
    model = SentenceTransformer(checkpoint)
    embeddings = model.encode(doc, batch_size=32, show_progress_bar=True)

    model_checkpoint = "sentence-transformers/" + checkpoint

    if search_type == "Semantic similarity (Embeddings)":
        return SearchFlatL2(checkpoint=model_checkpoint, sentence_embeddings=embeddings)
    elif search_type == "Word similarity (Jaccard)":
        return SearchJaccard(checkpoint=model_checkpoint, corpus=doc)


search_methods = ["Semantic similarity (Embeddings)", "Word similarity (Jaccard)"]


selected_search_method = st.sidebar.selectbox(
    "Similarity Metric",
    search_methods,
)

k = int(
    st.sidebar.number_input(
        label="Top n articles",
        min_value=1,
        value=10,
        step=1,
    )
)

input_text = st.text_area(
    "News text",
)


if input_text:
    searcher = load_search_methods(selected_search_method, checkpoint, df.copy())
    distances, indexes = searcher(
        input_text,
        k=k,
    )
    similar_df = df.iloc[indexes]
    with st.expander("Filter results"):

        date_filters = st.date_input(
            "Article dates",
            value=(similar_df["Published"].min(), similar_df["Published"].max()),
            min_value=similar_df["Published"].min(),
            max_value=similar_df["Published"].max(),
        )

        similar_df = similar_df[
            lambda df: df["Published"].dt.date.between(*date_filters)
        ].copy()

        # AND filter
        contains_all_words = st.text_input(
            f"Keep articles containing ALL keywords (separate by comma and space ', '):",
        ).lower()
        # ANY filter
        contains_any_words = st.text_input(
            f"Keep articles containing ANY keywords (separate by comma and space ', '):",
        ).lower()

        if len(contains_any_words) > 0:
            contains_any_list = contains_any_words.split(", ")
            similar_df = similar_df[
                lambda df: df["Headline"]
                .str.lower()
                .str.contains("|".join(contains_any_list))
            ]

        if len(contains_all_words) > 0:
            contains_all_list = contains_all_words.split(", ")
            text_column = similar_df["Headline"].str.lower()

            similar_df = similar_df[
                np.all([text_column.str.contains(t) for t in contains_all_list], axis=0)
            ]

    end_time = time.perf_counter()
    st.text(f"Took {end_time - start_time:.2f} seconds to search.")

    st.write(similar_df)

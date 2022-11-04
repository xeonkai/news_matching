import streamlit as st
from pathlib import Path
import pandas as pd
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss

st.set_page_config(layout="wide")

st.title("Article Similarity ❄️")
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


class SearchHNSW:
    """HNSW Approximate Nearest Neighbour L2 search, use when require low memory usage"""

    def __init__(
        self,
        checkpoint,
        sentence_embeddings,
        M=64,  # number of connections each vertex will have
        ef_search=32,  # depth of layers explored during search
        ef_construction=64,  # depth of layers explored during index construction
    ):
        self.model = SentenceTransformer(checkpoint)

        d = sentence_embeddings.shape[1]

        # HNSW config
        self.index = faiss.IndexHNSWFlat(d, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # faiss indexing embeddings
        self.index.add(sentence_embeddings)

    def __call__(self, text: str, k: int):  # class()

        xq = self.model.encode([text])
        D, I = self.index.search(xq, k)

        return D[0], I[0]


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


# TODO:
@st.cache()
class SearchCosine:
    """Exhaustive cosine similarity search on flat vector index"""

    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    # Cosine(u, v) = 1 - L2(u, v), u and v are normalized vectors == L2 on normalized vectors?
    pass


@st.cache()
def load_news_data(data_path):
    df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
    return df.copy(), df["title"].to_list(), df["content"].to_list()


@st.cache()
def load_title_embeddings(checkpoint):
    title_embeddings_path = Path(
        "data", "embeddings", f"{checkpoint}_title_embeddings.npy"
    )
    title_embeddings = np.load(title_embeddings_path)
    return title_embeddings


@st.cache()
def load_content_embeddings(checkpoint):
    content_embeddings_path = Path(
        "data", "embeddings", f"{checkpoint}_content_embeddings.npy"
    )
    content_embeddings = np.load(content_embeddings_path)
    return content_embeddings


data_path = Path("data", "processed", "sg_sanctions_on_russia.parquet")
df, titles, content = load_news_data(data_path)

title_or_content = st.sidebar.selectbox(
    "Compare Title or Content",
    (
        "Title",
        "Content",
    ),
)

# Set model
checkpoint = "multi-qa-MiniLM-L6-cos-v1"


@st.cache(allow_output_mutation=True)
def load_search_methods(search_type, checkpoint, title_or_content):
    # Fetch title or content embeddings
    embeddings = (
        load_title_embeddings(checkpoint)
        if title_or_content == "Title"
        else load_content_embeddings(checkpoint)
    )
    # Choose whether to compare titles or content
    doc = titles if title_or_content == "Title" else content

    model_checkpoint = "sentence-transformers/" + checkpoint
    # return function? or lazy class with attributes

    # TODO: save searchers and load saved versions here
    if search_type == "Fast Search":
        return SearchHNSW(checkpoint=model_checkpoint, sentence_embeddings=embeddings)
    elif search_type == "Exhaustive Search":
        return SearchFlatL2(checkpoint=model_checkpoint, sentence_embeddings=embeddings)
    elif search_type == "Jaccard":
        return SearchJaccard(checkpoint=model_checkpoint, corpus=doc)


search_methods = ["Fast Search", "Exhaustive Search", "Jaccard"]


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

article_id = st.text_input("Article ID")
if article_id:
    input_text = df.loc[
        article_id, "title" if title_or_content == "Title" else "content"
    ]
    with st.expander("Input Article"):
        st.markdown(input_text)
else:
    input_text = st.text_area(
        "News text",
    )


if input_text:
    searcher = load_search_methods(selected_search_method, checkpoint, title_or_content)
    distances, indexes = searcher(
        input_text,
        k=k,
    )
    similar_df = df.iloc[indexes]
    with st.expander("Filter results"):

        date_filters = st.date_input(
            "Article dates",
            value=(similar_df["date"].min(), similar_df["date"].max()),
            min_value=similar_df["date"].min(),
            max_value=similar_df["date"].max(),
        )

        similar_df = similar_df[
            lambda df: df["date"].dt.date.between(*date_filters)
        ].copy()
        filter_col1, filter_col2 = st.columns([1, 8])
        with filter_col1:
            filter_title_or_content = st.selectbox(
                "Filter based on title or content",
                (
                    "Title",
                    "Content",
                ),
            )
        with filter_col2:
            # OR filter
            contains_any_words = st.text_input(
                f"{filter_title_or_content} can contain any of: (separate with comma and space ', ')",
            ).lower()

            # AND filter
            contains_all_words = st.text_input(
                f"{filter_title_or_content} must contain all of: (separate with comma and space ', ')",
            ).lower()

        if len(contains_any_words) > 0:
            contains_any_list = contains_any_words.split(", ")
            similar_df = similar_df[
                lambda df: df[
                    "title" if filter_title_or_content == "Title" else "content"
                ]
                .str.lower()
                .str.contains("|".join(contains_any_list))
            ]

        if len(contains_all_words) > 0:
            contains_all_list = contains_all_words.split(", ")
            text_column = similar_df[
                "title" if filter_title_or_content == "Title" else "content"
            ].str.lower()

            similar_df = similar_df[
                np.all([text_column.str.contains(t) for t in contains_all_list], axis=0)
            ]

    end_time = time.perf_counter()
    st.text(f"Took {end_time - start_time:.2f} seconds to search.")

    st.write(
        similar_df[
            [
                "title",
                "content",
                "url",
                "date",
            ]
        ]
    )

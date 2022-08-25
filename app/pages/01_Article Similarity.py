import streamlit as st
from pathlib import Path
import pandas as pd
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pynndescent import NNDescent

st.set_page_config(layout="wide")

st.title("Article Similarity ❄️")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()


class SearchJaccard:
    """Jaccard similarity based on tokenized word sets"""

    def __init__(
        self,
        model,
        corpus: list[str],
    ):
        self.tokenizer = model.tokenizer

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


@st.cache()
class SearchHNSW:
    """HNSW Approximate Nearest Neighbour L2 search, use when require low memory usage"""

    def __init__(
        self,
        model,
        sentence_embeddings,
        M=64,  # number of connections each vertex will have
        ef_search=32,  # depth of layers explored during search
        ef_construction=64,  # depth of layers explored during index construction
    ):
        self.model = model

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


@st.cache()
class SearchFlatL2:
    """Exhaustive euclidean search on flat vector index"""

    def __init__(
        self,
        model,
        sentence_embeddings,
    ):
        self.model = model

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
class SearchCosine:
    """Exhaustive cosine similarity search on flat vector index"""

    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    # Cosine(u, v) = 1 - L2(u, v), u and v are normalized vectors == L2 on normalized vectors?
    pass


@st.cache()
class SearchPyNN:
    """NNDescent"""

    def __init__(
        self,
        model,
        sentence_embeddings,
    ):
        self.model = model

        # PyNNDescent indexing embeddings
        self.index = NNDescent(sentence_embeddings)

    def __call__(self, text: str, k: int):

        xq = self.model.encode([text])
        I, D = self.index.query(xq, k)

        return D[0], I[0]


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


# TODO: Update to load only one model at a time
@st.cache(allow_output_mutation=True)
def load_model(checkpoint):
    if checkpoint == "doc2vec":
        return NotImplementedError
    return SentenceTransformer(checkpoint)


data_path = Path("data", "processed", "sg_sanctions_on_russia.parquet")
df, titles, content = load_news_data(data_path)

model_checkpoints = {
    "MiniLM": "multi-qa-MiniLM-L6-cos-v1",
    "Doc2vec": "",
    "MPNet": "all-mpnet-base-v2",
}


title_or_content = st.sidebar.selectbox(
    "Compare Title or Content",
    (
        "Title",
        "Content",
    ),
)


st.sidebar.warning("In progress")

# Choose model
selected_model = st.sidebar.selectbox("Model Type", model_checkpoints.keys())
checkpoint = model_checkpoints[selected_model]
model = load_model(checkpoint)

# Choose whether to compare titles or content
doc = titles if title_or_content == "Title" else "content"
# Fetch title or content embeddings
embeddings = (
    load_title_embeddings(checkpoint)
    if title_or_content == "Title"
    else load_content_embeddings(checkpoint)
)

# TODO: Fix cache errors
search_methods = {
    "Fast Search": SearchHNSW(model=model, sentence_embeddings=embeddings),
    # "Faster": SearchPyNN(model=model, sentence_embeddings=embeddings), # TODO: Problem with hashing class
    "Exhaustive Search": SearchFlatL2(model=model, sentence_embeddings=embeddings),
    # "Jaccard": SearchJaccard(model=model, corpus=doc), # TODO: Problem with hashing class
    # "Cosine": SearchCosine(model=model, sentence_embeddings=embeddings),
}

selected_search_method = st.sidebar.selectbox(
    "Similarity Metric",
    search_methods.keys(),
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

searcher = search_methods[selected_search_method]
distances, indexes = searcher(
    input_text,
    k=k,
)

similar_df = df.iloc[indexes]

end_time = time.perf_counter()
st.text(f"Took {end_time - start_time:.2f} seconds to search.")

st.write(
    similar_df[
        [
            "title",
            "content",
            "url",
        ]
    ]
)

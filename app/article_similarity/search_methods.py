from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss
import pandas as pd
import streamlit as st


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


@st.cache(allow_output_mutation=True)
def load_similarity_search_methods(search_type, checkpoint, df):
    doc = df["Headline"].to_list()
    model = SentenceTransformer(checkpoint)
    embeddings = model.encode(doc, batch_size=32, show_progress_bar=True)

    model_checkpoint = "sentence-transformers/" + checkpoint

    if search_type == "Semantic similarity (Embeddings)":
        return SearchFlatL2(checkpoint=model_checkpoint, sentence_embeddings=embeddings)
    elif search_type == "Word similarity (Jaccard)":
        return SearchJaccard(checkpoint=model_checkpoint, corpus=doc)


# Corresponds to selection above
SIM_SEARCH_METHODS = ("Semantic similarity (Embeddings)", "Word similarity (Jaccard)")

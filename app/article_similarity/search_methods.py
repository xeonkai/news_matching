from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st


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
def load_similarity_search(checkpoint, text_ls):
    doc = text_ls.copy()
    model = SentenceTransformer(checkpoint)
    embeddings = model.encode(doc, batch_size=32, show_progress_bar=True)

    model_checkpoint = "sentence-transformers/" + checkpoint

    return SearchFlatL2(checkpoint=model_checkpoint, sentence_embeddings=embeddings)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []

    def create_index(self, texts):

        if not texts:
            raise ValueError("No text provided")

        self.text_chunks = texts

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        if len(embeddings.shape) < 2:
            embeddings = embeddings.reshape(1,-1)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query, top_k=3):

        if self.index is None:
            return []

        q = self.model.encode([query])
        q = np.array(q).astype("float32")

        distances, indices = self.index.search(q, top_k)

        results=[]
        for idx in indices[0]:
            if 0 <= idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

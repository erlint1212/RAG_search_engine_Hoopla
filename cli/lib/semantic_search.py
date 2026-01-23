#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import numpy as np
import os
from constants import *

class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self._cur_path = os.path.dirname(__file__)
        self._top_path = os.path.join(self._cur_path, "..", "..")
        self._cache_path = os.path.join(self._top_path, "cache")
        self._embeddings_path = os.path.join(self._cache_path, "movie_embeddings.npy")

    def search(self, query  : str, limit : int = LIMIT) -> list[tuple[float, dict[int, str, str]]]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        embedded_query = self.generate_embedding(query)

        simmilarity_list = []
        for i, embedded_doc in enumerate(self.embeddings): 
            cos_sim = cosine_similarity(embedded_doc, embedded_query)

            simmilarity_list.append((cos_sim, self.documents[i]))

        simmilarity_list.sort(key=lambda x: x[0], reverse=True)

        result_dic = []
        for i in range(limit):
            result_dic.append(
                        {
                            "score" : simmilarity_list[i][0],
                            "title" : simmilarity_list[i][1]["title"],
                            "description" : simmilarity_list[i][1]["description"],
                        }
                    )

        return result_dic



    def build_embeddings(self, documents : list[dict[int, list[int | str]]]) -> list[float]:
        self.documents = documents
        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")

        encoded = self.encode(doc_list)

        self.embeddings = encoded

        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

        with open(self._embeddings_path, "wb") as embeddings_file:
            np.save(embeddings_file, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents : list[dict[int, list[int | str]]]) -> list[float]:

        if os.path.exists(self._embeddings_path):

            self.documents = documents

            with open(self._embeddings_path, "rb") as embeddings_file:
                self.embeddings = np.load(embeddings_file)

            return self.embeddings
        else:
            return self.build_embeddings(documents)

    def encode(self, text : list[str]) -> list[float]:
        encoded_text = self.model.encode(text, show_progress_bar=True)

        return encoded_text

    def generate_embedding(self, text : str) -> list[float]:
        clean_text = text.strip()
        if clean_text == "":
            ValueError("The input text is empty")

        embedding = self.encode(text)

        return embedding


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def cosine_similarity(vec1 : list[float], vec2 : list[float]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

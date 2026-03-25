import json
import os

import numpy as np
from constants import *
from sentence_transformers import SentenceTransformer

from . import semantic_search as semsearch


class ChunkedSemanticSearch(semsearch.SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = []
        self._embeddings_path = os.path.join(self._cache_path, "chunk_embeddings.npy")
        self._metadata_path = os.path.join(self._cache_path, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> None:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []

        for doc_idx, doc in enumerate(self.documents):
            if doc["description"] == "":
                continue

            chunks = semsearch.semantic_chunk(
                text_block=doc["description"], max_chunk_size=4, overlap=1
            )

            all_chunks += chunks

            for chunk_idx in range(len(chunks)):
                self.chunk_metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )

        encoded = self.encode(all_chunks)

        self.chunk_embeddings = encoded

        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

        with open(self._embeddings_path, "wb") as embeddings_file:
            np.save(embeddings_file, self.chunk_embeddings)

        with open(self._metadata_path, "w") as metadata_file:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                metadata_file,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(self._embeddings_path) and os.path.exists(
            self._metadata_path
        ):

            self.documents = documents

            with open(self._embeddings_path, "rb") as embeddings_file:
                self.chunk_embeddings = np.load(embeddings_file)

            with open(self._metadata_path, "r") as metadata_file:
                self.chunk_metadata = json.load(metadata_file)["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        encoded_query = self.encode(query)
        chunk_score_list = []

        for chunk_idx, chunk in enumerate(self.chunk_embeddings):
            score = semsearch.cosine_similarity(encoded_query, chunk)

            chunk_score_list.append(
                    {
                        "chunk_idx" : chunk_idx,
                        "movie_idx" : self.chunk_metadata[chunk_idx]["movie_idx"],
                        "score" : score,
                    }
            )

        mov_score_map = {}
        for chunk in chunk_score_list:
            if chunk["movie_idx"] not in mov_score_map.keys() or chunk["score"] > mov_score_map[chunk["movie_idx"]]:
                mov_score_map[chunk["movie_idx"]] = chunk["score"]

        sorted_items = sorted(mov_score_map.items(), key=lambda item: item[1], reverse=True)[:limit]
        sorted_mov_score_map = dict(sorted_items)

        return_list = []
        for movie_idx, score in sorted_mov_score_map.items():
            doc = self.documents[movie_idx]
            return_list.append(
                    {
                        "id" : movie_idx,
                        "title" : doc["title"],
                        "document" : doc["description"][:100],
                        "score" : round(score, SCORE_PRECISION),
                        "metadata" : doc.get("metadata") or {},
                    }
            )

        return return_list


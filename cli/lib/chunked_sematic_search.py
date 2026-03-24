import json
import os

import numpy as np
from constants import *
from sentence_transformers import SentenceTransformer

from .semantic_search import SemanticSearch, semantic_chunk


class ChunkedSemanticSearch(SemanticSearch):
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

            chunks = semantic_chunk(
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
                self.chunk_metadata = json.load(metadata_file)

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

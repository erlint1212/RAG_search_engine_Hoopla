from semantic_search.py import SemanticSearch
from ../semantic_search_cli.py import semantic_chunk

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents) -> None:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        chunk_list = []
        chunk_metadata = {}

        for i, doc in enumerate(self.documents):
            if doc['description'] == "":
                continue
            
            chunks = semantic_chunk(text_block=doc['description'], max_chunk_size=4, overlap=1)

            chunk_list.append(chunks)

            chunk_metadata[chunks] = {
                    doc["id"], 
                    i, 
                    len(chunks)
                }


        encoded = self.encode(doc_list)

        self.embeddings = encoded

        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

        with open(self._embeddings_path, "wb") as embeddings_file:
            np.save(embeddings_file, self.embeddings)

        return self.embeddings

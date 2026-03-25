import os

from constants import *

from .keyword_search import InvertedIndex
from .chunked_sematic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()

        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()


    def _bm25_search(self, query : str, limit : int) -> list[float]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query : str, alpha: float, limit : int = LIMIT) -> list[float]:
        # id : score
        bm25_dic = self._bm25_search(query, limit * 500)
        semsearch_dic = self.semantic_search.search_chunks(query, limit * 500)
        sem_score_dict = {d["id"]: d["score"] for d in semsearch_dic}
        
        bm25_dic_norm = normalize_dict(bm25_dic)
        sem_score_dic_norm = normalize_dict(sem_score_dict)

        comb_score_dic = {}
        for movie_id, norm_score in bm25_dic_norm:
            comb_score_dic[movie_id] = {"keyword_score" : norm_score, "semantic_score": 0}
        for movie_id, norm_score in sem_score_dic_norm:
            if movie_id not in comb_score_dic.keys():
                comb_score_dic[movie_id] = {"keyword_score" : 0, "semantic_score": norm_score}
                continue
            comb_score_dic[movie_id]["semantic_score"] = norm_score

        for movie_id in comb_score_dic.keys():
            scores = comb_score_dic[movie_id]
            scores["hybrid_score"] = hybrid_score(scores["keyword_score"], scores["semantic_score"], alpha)

        comb_score_dic_sorted = dict(
            sorted(comb_score_dic.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)[:limit]
        )

        return comb_score_dic_sorted






    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize(score_list : list[float]) -> list[float]:
    max_score = max(score_list)
    min_score = min(score_list)

    if min_score == max_score:
        return [1.0] * len(score_list)

    norm_list = [(score - min_score) / (max_score - min_score) for score in score_list]

    return norm_list

def normalize_dict(score_dict: dict[int, float]) -> dict[int, float]:
    scores = list(score_dict.values())
    normed = normalize(scores)
    return dict(zip(score_dict.keys(), normed))

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

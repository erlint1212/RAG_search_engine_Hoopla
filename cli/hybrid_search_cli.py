import argparse
import lib.hybrid_search as hybrid_search

import os
import json

from constants import *

def normalize(score_list : list[float]) -> list[float]:
    return hybrid_search.normalize(score_list)

def weighted_search(query : str, alpha : float = DEFAULT_ALPHA, limit : int = LIMIT):

    cur_path = os.path.dirname(__file__)
    movie_path = os.path.join(cur_path, "..", "data", "movies.json") 

    with open(movie_path, "r") as mov_file:
        movies = json.load(mov_file)["movies"]

    hybrid_class = hybrid_search.HybridSearch(movies)
    sorted_scores = hybrid_class.weighted_search(query, alpha, limit)
    
    document_map = {}
    for doc in hybrid_class.documents:
        document_map[doc["id"]] = doc

    for rank, (movie_id, scores) in enumerate(sorted_scores.items(), start=1):
        doc = document_map[movie_id]
        print(f"{rank}. {doc["title"]}")
        print(f"   Hybrid Score: {scores["hybrid_score"]:.4f}")
        print(f"   BM25: {scores["keyword_score"]:.4f}, Semantic: {scores["semantic_score"]:.4f}")
        print(f"   {doc["description"][:100]}...")

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_command = subparsers.add_parser("normalize", help="Normalize a list of floats")
    normalize_command.add_argument("score_list", type=float, nargs="+", help="List of floats to normalize")

    weighted_search_command = subparsers.add_parser("weighted-search", help="Normalize a list of floats")
    weighted_search_command.add_argument("query", type=str, help="Query for searching")
    weighted_search_command.add_argument( '--alpha', type=float, default=DEFAULT_ALPHA, help="Optional: alpha (or \"α\") is just a constant that we can use to dynamically control the weighting between the two scores.")
    weighted_search_command.add_argument( '--limit', type=int, default=LIMIT, help="Optional: set a limit on the number of items to process.")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_list = normalize(args.score_list)
            for score in norm_list:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

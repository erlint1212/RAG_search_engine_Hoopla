#!/usr/bin/env python3

import argparse
import os
import json
import string
from nltk.stem import PorterStemmer

from inverted_index import InvertedIndex

def build() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    docs = inverted_index.get_documents("merida")

    print(f"First document for token 'merida' = {docs[0]}")

def clean(dirty_str : str, stop_words : list[str]) -> list[str]:
    cleaned_str = dirty_str.lower() # Case senesetive
    cleaned_str = cleaned_str.translate(str.maketrans('', '', string.punctuation)) # Remove punctuations
    cleaned_str = cleaned_str.split(" ") # Tokenization
    cleaned_str = [word for word in cleaned_str if word not in stop_words] # Remove stop words

    stemmer = PorterStemmer()
    cleaned_str = [stemmer.stem(token) for token in cleaned_str] # Stem words

    return cleaned_str

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def search_command(query : str, limit : int = 5) -> list[dict]:
    print(f"Searching for: {query}")

    cur_path = os.path.dirname(__file__)
    print(cur_path)
    data_mov_path = os.path.join(cur_path, "..", "data", "movies.json")
    print(data_mov_path)

    with open(os.path.join(cur_path, "..", "data", "stopwords.txt"), "r") as stop_words_file:
        stop_words = stop_words_file.read()
        stop_words = stop_words.splitlines()

    cleaned_query = clean(query, stop_words)
    
    with open(data_mov_path, "r") as mov_file:
        mov_dict = json.load(mov_file)

    movie_matches = []
    for movie in mov_dict["movies"]:
        cleaned_title = clean(movie["title"], stop_words)
        if has_matching_token(cleaned_query, cleaned_title):
        #if any(map(lambda v: v in cleaned_query, cleaned_title)):
            movie_matches.append(movie)

    return movie_matches


def main() -> None:
    trunc_len = 5

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build and inverted index and save it to file")

    args = parser.parse_args()

    match args.command:
        case "search":

            movie_matches = search_command(args.query, trunc_len)

            def idSort(e):
                return e["id"]
            movie_matches.sort(key=idSort, reverse=False)

            list_len = trunc_len if trunc_len < len(movie_matches) else len(movie_matches)
            for i in range(list_len):
                print(f"{i+1}. " + movie_matches[i]["title"])

            if len(movie_matches) > trunc_len:
                print("...")
            
            print(f"Total found: {len(movie_matches)}")

        case "build":
            build()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import os
import json
import string
from nltk.stem import PorterStemmer

from inverted_index import InvertedIndex

def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    #docs = inverted_index.get_documents("merida")

    #print(f"First document for token 'merida' = {docs[0]}")

def tf_command(doc_id : int, term : str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()

    term_freq = inverted_index.get_tf(doc_id, term)

    return term_freq

def idf_command(term : str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    idf_val = inverted_index.get_idf(term)

    return idf_val

def tf_idf_command(doc_id : int, term : str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    tf_idf = inverted_index.get_tfidf(doc_id, term)

    return tf_idf

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

def search_command(query : str, limit : int = 5) -> tuple[list[dict], bool, int]:
    print(f"Searching for: {query}")

    inverted_index = InvertedIndex()
    inverted_index.load()

    tokens = clean(query, inverted_index.get_stopwords())
    
    movie_matches = []

    
    doc_id_matches = [] 
    for token in tokens:
        doc_id_matches += inverted_index.get_documents(token, limit)

    total_matches_found = len(doc_id_matches)
    for i, doc_id in enumerate(doc_id_matches):
        if i > limit - 1:
            break
        movie_matches.append(inverted_index.docmap[doc_id])

    movie_matches.sort(key=lambda m: m["id"], reverse=False)

    over_limit = True if len(doc_id_matches) > limit else False

    return movie_matches, over_limit, total_matches_found


def main() -> None:
    trunc_len = 5

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and inverted index and save it to file")

    tf_parser = subparsers.add_parser("tf", help="Search term frequency in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term which you want frequency for")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term which you want inverse frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Search term tfidf for a document and doc id")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term which you want frequency for")

    args = parser.parse_args()

    match args.command:
        case "search":
            movie_matches, over_limit, total_matches_found = search_command(args.query, trunc_len)

            for i in range(len(movie_matches)):
                print(f"{i+1}. " + movie_matches[i]["title"])

            if over_limit:
                print("...")
            
            print(f"Total found: {total_matches_found}")

        case "build":
            build_command()

        case "tf":
            term_frequency = tf_command(args.doc_id, args.term)

            print(f'Doc ID : {args.doc_id}, Term: "{args.term}", Frequency: "{term_frequency}"')

        case "idf":
            idf = idf_command(args.term)

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

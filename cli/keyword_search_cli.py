#!/usr/bin/env python3

import argparse
import os
import json
import string

stop_words = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", "few", "for", "from", "further", 
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", 
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", 
    "me", "more", "most", "mustn't", "my", "myself", 
    "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
    "under", "until", "up", 
    "very", 
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", 
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    ]

def clean(dirty_str : str) -> str:
    cleaned_str = dirty_str.lower() # Case senesetive
    cleaned_str = cleaned_str.translate(str.maketrans('', '', string.punctuation)) # Remove punctuations
    cleaned_str = cleaned_str.split(" ") # Tokenization
    #cleaned_str = [word for word in cleaned_str if word not in stop_words] # Remove stop words
    return cleaned_str


def main() -> None:
    trunc_len = 5
    cur_path = os.path.dirname(__file__)
    print(cur_path)
    data_mov_path = os.path.join(cur_path, "..", "data", "movies.json")
    print(data_mov_path)

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            cleaned_query = clean(args.query)
            
            with open(data_mov_path, "r") as mov_file:
                mov_dict = json.load(mov_file)

            movie_matches = []
            for movie in mov_dict["movies"]:
                cleaned_title = clean(movie["title"])
                if any(map(lambda v: v in cleaned_query, cleaned_title)):
                    movie_matches.append(movie)

            def idSort(e):
                return e["id"]
            movie_matches.sort(key=idSort, reverse=False)

            list_len = trunc_len if trunc_len < len(movie_matches) else len(movie_matches)
            for i in range(list_len):
                print(f"{i+1}. " + movie_matches[i]["title"])

            if len(movie_matches) > trunc_len:
                print("...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

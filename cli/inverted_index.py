#!/usr/bin/env python3

import pickle
import argparse
import os
import json
import string
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.cur_path = os.path.dirname(__file__)
        self.data_mov_path = os.path.join(self.cur_path, "..", "data", "movies.json")
        self.stopwords_path = os.path.join(self.cur_path, "..", "data", "stopwords.txt")
        self.cache_path = os.path.join(self.cur_path, "..", "cache")

    def __add_document(self, doc_id : int, text : str):
        cur_path = os.path.dirname(__file__)

        with open(os.path.join(cur_path, "..", "data", "stopwords.txt"), "r") as stop_words_file:
            stop_words = stop_words_file.read()
            stop_words = stop_words.splitlines()

        cleaned_tokens = self.__tokenize(text, stop_words)

        for token in cleaned_tokens:
            try:
                self.index[token].add(doc_id)
            except KeyError:
                self.index[token] = {doc_id}

    def get_documents(self, term : str) -> list[int]:
        doc_id_matches = []

        term = term.lower()
        title_tokens = self.index.keys()

        if term in self.index:
            doc_id_matches = self.index[term]
        else:
            doc_id_matches = set()

        doc_id_matches = sorted(set(doc_id_matches), reverse=False) # Remove duplicates

        return doc_id_matches

    def build(self) -> None:

        with open(self.data_mov_path, "r") as mov_file:
            movies = json.load(mov_file)

        for movie in movies["movies"]:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie # Doubble saving of id?

    def save(self) -> None:

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        with open(os.path.join(self.cache_path, "index.pkl"), "wb") as index_file:
            pickle.dump(self.index, index_file)

        with open(os.path.join(self.cache_path, "docmap.pkl"), "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

    def __tokenize(self, dirty_str : str, stop_words : list[str]) -> list[str]:
        cleaned_str = dirty_str.lower() # Case senesetive
        cleaned_str = cleaned_str.translate(str.maketrans('', '', string.punctuation)) # Remove punctuations
        cleaned_str = cleaned_str.split(" ") # Tokenization
        cleaned_str = [word for word in cleaned_str if word not in stop_words] # Remove stop words

        stemmer = PorterStemmer()
        cleaned_str = [stemmer.stem(token) for token in cleaned_str] # Stem words

        return cleaned_str

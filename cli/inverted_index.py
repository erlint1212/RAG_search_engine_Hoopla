#!/usr/bin/env python3

import pickle
import math
import argparse
import os
import json
import string
from nltk.stem import PorterStemmer
from collections import Counter

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {} # doc_id : Counter objects
        self._cur_path = os.path.dirname(__file__)
        self._data_mov_path = os.path.join(self._cur_path, "..", "data", "movies.json")
        self._stopwords_path = os.path.join(self._cur_path, "..", "data", "stopwords.txt")
        self._cache_path = os.path.join(self._cur_path, "..", "cache")
        self._index_path = os.path.join(self._cache_path, "index.pkl")
        self._docmap_path = os.path.join(self._cache_path, "docmap.pkl")
        self._term_frequencies_path = os.path.join(self._cache_path, "term_frequencies.pkl")

    def get_stopwords(self) -> list[str]:

        with open(os.path.join(self._cur_path, "..", "data", "stopwords.txt"), "r") as stop_words_file:
            stop_words = stop_words_file.read()
            stop_words = stop_words.splitlines()

        return stop_words

    def __add_document(self, doc_id : int, text : str, stop_words : list[str]) -> None:

        cleaned_tokens = self.__tokenize(text, stop_words)

        self.term_frequencies[doc_id] = Counter()

        for token in cleaned_tokens:
            try:
                self.index[token].add(doc_id)
            except KeyError:
                self.index[token] = {doc_id}

            self.term_frequencies[doc_id][token] += 1

    def get_tf(self, doc_id : int, term : str) -> int:

        token = self.__tokenize(term, self.get_stopwords)
        if len(token) > 1:
            raise Exception(f"Expected one term, got multiple: {token}")

        token = token[0]

        try:
            term_freq = self.term_frequencies[doc_id][token] 
        except KeyError:
            term_freq = 0

        return term_freq

    def get_idf(self, term : str) -> float:

        token = self.__tokenize(term, self.get_stopwords)
        if len(token) > 1:
            raise Exception(f"Expected one term, got multiple: {token}")

        token = token[0]

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id : int, term : str) -> float:

        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf*idf

    def get_bm25_idf(self, term: str) -> float:

        token = self.__tokenize(term, self.get_stopwords)
        if len(token) > 1:
            raise Exception(f"Expected one term, got multiple: {token}")

        token = token[0]
        
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])

        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_documents(self, term : str, limit : int = 0) -> list[int]:
        doc_id_matches = []

        term = term.lower()
        title_tokens = self.index.keys()
        
        
        """
        # Search inside each token
        doc_id_matches = []
        for index_token in self.index.keys():
            if term in index_token:
                doc_id_matches += self.index[index_token]
                if limit != 0:
                    if len(doc_id_matches) > limit:
                        break
        """

        if term in self.index:
            doc_id_matches = self.index[term]
        else:
            doc_id_matches = set()

        doc_id_matches = sorted(set(doc_id_matches), reverse=False) # Remove duplicates

        return doc_id_matches

    def build(self) -> None:

        with open(self._data_mov_path, "r") as mov_file:
            movies = json.load(mov_file)

        stop_words = self.get_stopwords()

        for movie in movies["movies"]:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}", stop_words)
            self.docmap[movie["id"]] = movie # Doubble saving of id?

    def save(self) -> None:

        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

        with open(self._index_path, "wb") as index_file:
            pickle.dump(self.index, index_file)

        with open(self._docmap_path, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open(self._term_frequencies_path, "wb") as term_frequencies_file:
            pickle.dump(self.term_frequencies, term_frequencies_file)

    def load(self) -> None:

        if not os.path.exists(self._cache_path):
            raise FileNotFoundError(f"cache path not found: {self._cache_path}")
        if not os.path.exists(self._index_path):
            raise FileNotFoundError(f"index.pkl file not found: {self._index_path}")
        if not os.path.exists(self._docmap_path):
            raise FileNotFoundError(f"docmap.pkl file not found: {self._docmap_path}")
        if not os.path.exists(self._term_frequencies_path):
            raise FileNotFoundError(f"term_frequencies.pkl file not found: {self._term_frequencies_path}")
        
        with open(self._index_path, "rb") as index_file:
            self.index = pickle.load(index_file)

        with open(self._docmap_path, "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)

        with open(self._term_frequencies_path, "rb") as term_frequencies_file:
            self.term_frequencies = pickle.load(term_frequencies_file)

    def __tokenize(self, dirty_str : str, stop_words : list[str]) -> list[str]:

        cleaned_str = dirty_str.lower() # Case senesetive
        cleaned_str = cleaned_str.translate(str.maketrans('', '', string.punctuation)) # Remove punctuations
        cleaned_str = cleaned_str.split(" ") # Tokenization

        stop_words = self.get_stopwords()
        cleaned_str = [word for word in cleaned_str if word not in stop_words] # Remove stop words

        stemmer = PorterStemmer()
        cleaned_str = [stemmer.stem(token) for token in cleaned_str] # Stem words

        return cleaned_str

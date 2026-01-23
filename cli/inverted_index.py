#!/usr/bin/env python3

import pickle
import math
import argparse
import os
import json
import string
from nltk.stem import PorterStemmer
from collections import Counter
from itertools import islice

from constants import *

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {} # doc_id : Counter objects
        self.doc_lengths = {} # doc_id : length of tokens
        self._cur_path = os.path.dirname(__file__)
        self._data_mov_path = os.path.join(self._cur_path, "..", "data", "movies.json")
        self._stopwords_path = os.path.join(self._cur_path, "..", "data", "stopwords.txt")
        self._cache_path = os.path.join(self._cur_path, "..", "cache")
        self._index_path = os.path.join(self._cache_path, "index.pkl")
        self._docmap_path = os.path.join(self._cache_path, "docmap.pkl")
        self._term_frequencies_path = os.path.join(self._cache_path, "term_frequencies.pkl")
        self._doc_lengths_path = os.path.join(self._cache_path, "doc_lengths.pkl")

    def __get_stopwords(self) -> list[str]:

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

        self.doc_lengths[doc_id] = len(cleaned_tokens)

    def get_tf(self, doc_id : int, term : str) -> int:

        stop_words = self.__get_stopwords()
        token = self.__tokenize(term, stop_words)
        if len(token) > 1:
            raise Exception(f"Expected one term, got multiple: {token}")

        token = token[0]

        try:
            term_freq = self.term_frequencies[doc_id][token] 
        except KeyError:
            term_freq = 0

        return term_freq

    def get_idf(self, term : str) -> float:

        stop_words = self.__get_stopwords()
        token = self.__tokenize(term, stop_words)
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

        stop_words = self.__get_stopwords()
        token = self.__tokenize(term, stop_words)
        if len(token) > 1:
            raise Exception(f"Expected one term, got multiple: {token}")

        token = token[0]
        
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])

        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id : int, term : str, k1 : float = BM25_K1, b : float = BM25_B) -> float:

        tf = self.get_tf(doc_id, term)

        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id]/ avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_bm25(self, doc_id : int, term : str) -> float:

        bm25_tf = self.get_bm25_tf(doc_id, term) 
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query : str, limit : int = 5):

        stop_words = self.__get_stopwords()
        tokens = self.__tokenize(query, stop_words)

        scores = {} # doc_id : BM25 cost
        for token in tokens:
            for doc_id in self.index[token]:
                try:
                    scores[doc_id] += self.get_bm25(doc_id, token)
                except KeyError:
                    scores[doc_id] = self.get_bm25(doc_id, token)
        
        #scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit])

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

        stop_words = self.__get_stopwords()

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

        with open(self._doc_lengths_path, "wb") as doc_lengths_file:
            pickle.dump(self.doc_lengths, doc_lengths_file)

    def load(self) -> None:

        paths = [self._cache_path, self._index_path, self._docmap_path, self._term_frequencies_path, self._doc_lengths_path]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Load path not found: {path}")
        
       
        with open(self._index_path, "rb") as index_file:
            self.index = pickle.load(index_file)

        with open(self._docmap_path, "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)

        with open(self._term_frequencies_path, "rb") as term_frequencies_file:
            self.term_frequencies = pickle.load(term_frequencies_file)

        with open(self._doc_lengths_path, "rb") as doc_lengths_file:
            self.doc_lengths = pickle.load(doc_lengths_file)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) <= 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def __tokenize(self, dirty_str : str, stop_words : list[str]) -> list[str]:

        cleaned_str = dirty_str.lower() # Case senesetive
        cleaned_str = cleaned_str.translate(str.maketrans('', '', string.punctuation)) # Remove punctuations
        cleaned_str = cleaned_str.split(" ") # Tokenization

        cleaned_str = [word for word in cleaned_str if word not in stop_words] # Remove stop words

        stemmer = PorterStemmer()
        cleaned_str = [stemmer.stem(token) for token in cleaned_str] # Stem words

        return cleaned_str

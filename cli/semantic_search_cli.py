#!/usr/bin/env python3

import argparse
import lib.semantic_search as semsearch
import os
import json
from constants import *
import re

def verify_command() -> None:
    semsearch.verify_model()

def embed_text(text : str) -> None:
    semantic_search = semsearch.SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = semsearch.SemanticSearch()
    cur_path = os.path.dirname(__file__)
    movie_path = os.path.join(cur_path, "..", "data", "movies.json") 

    with open(movie_path, "r") as mov_file:
        movies = json.load(mov_file)

    embeddings = semantic_search.load_or_create_embeddings(movies["movies"])

    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query : str):
    semantic_search = semsearch.SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def search(query : str, limit : int  = LIMIT) -> None:
    semantic_search = semsearch.SemanticSearch()
    cur_path = os.path.dirname(__file__)
    movie_path = os.path.join(cur_path, "..", "data", "movies.json") 

    with open(movie_path, "r") as mov_file:
        movies = json.load(mov_file)

    embeddings = semantic_search.load_or_create_embeddings(movies["movies"])

    doc_dic = semantic_search.search(query, limit)

    for i, doc in enumerate(doc_dic): 
        print(f"{i+1}. {doc["title"]} (score: {doc["score"]:.4f})")
        print(f"   {doc["description"][:100]}...")
        print("")

def chunk(text_block : str, chunk_size : int = CHUNK_SIZE, overlap : int = 0) -> None:
    if overlap < 0 or type(overlap) != int:
        raise ValueError("overlap must be 0 or a positive integer")
    words = text_block.split(" ")
    #chunks = [text_block[i:i + chunk_size] for i in range(0, len(text_block), chunk_size)]
    chunks = []
    for i in range(0, len(words), chunk_size):
        if i == 0:
            chunk = (words[i:i + chunk_size])
        else:
            chunk = (words[i - overlap:i + chunk_size])
        chunk = " ".join(chunk)
        chunks.append(chunk)

    print(f"Chunking {len(text_block)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def semantic_chunk(text_block : str, chunk_size : int = CHUNK_SIZE, overlap : int = 0) -> None:
    if overlap < 0 or type(overlap) != int:
        raise ValueError("overlap must be 0 or a positive integer")
    sentences = re.split(r"(?<=[.!?])\s+", text_block)
    #chunks = [text_block[i:i + chunk_size] for i in range(0, len(text_block), chunk_size)]
    chunks = []
    temp_sentence = ""
    for sentence in sentences:
        if sentence >= max_chunk_size:
            chunks.append(sentence)
            continue
        temp_sentence += sentence
        if temp_sentence >= max_chunk_size:
            chunks.append(temp_sentence)
            temp_sentence = ""

    for i in range(0, len(words), chunk_size):
        if i == 0:
            chunk = (words[i:i + chunk_size])
        else:
            chunk = (words[i - overlap:i + chunk_size])
        chunk = " ".join(chunk)
        chunks.append(chunk)

    print(f"Chunking {len(text_block)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Test if the model works")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embedded text")
    embed_text_parser.add_argument("text", type=str, help="String to be embedded")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify that the loading works")

    embed_query_parser = subparsers.add_parser("embedquery", help="Command that accepts a positional query string argument. It should call your embed_query_text function with the provided query.")
    embed_query_parser.add_argument("query", type=str, help="String to be embedded")

    search_parser = subparsers.add_parser("search", help="Command that accepts a positional query string argument. It should call your embed_query_text function with the provided query.")
    search_parser.add_argument("query", type=str, help="String to be embedded")
    search_parser.add_argument(
        '--limit',
        nargs="?",
        type=int,
        default=LIMIT,
        help="Specify the maximum number of items to print (e.g., --limit 10)"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Command that accepts a positional query string argument. It should call your embed_query_text function with the provided query.")
    chunk_parser.add_argument("text_block", type=str, help="String to chunk")
    chunk_parser.add_argument(
        '--chunk-size',
        nargs="?",
        type=int,
        default=CHUNK_SIZE,
        help="Specify the maximum number of items to print (e.g., --limit 10)"
    )
    chunk_parser.add_argument(
        '--overlap',
        type=int,
        default=0,
        help="Specify the maximum number of items to overlap (e.g., --limit 10)"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_command()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text_block, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

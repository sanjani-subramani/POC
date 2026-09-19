"""Minimal vector memory POC: local embeddings + ChromaDB persistent storage."""
import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.expanduser("~"), "POC", "chroma_data")
MODEL_NAME = "all-MiniLM-L6-v2"


def cosine_similarity(vec_a, vec_b):
    a, b = np.asarray(vec_a, dtype=float), np.asarray(vec_b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    # cosine space => Chroma distance = 1 - cosine similarity
    notes = client.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})
    print(f"Loaded collection 'notes' with {notes.count()} stored notes.")
    print("Commands: add <text> | query <text> | quit")

    while True:
        line = input("\n> ").strip()
        cmd, _, text = line.partition(" ")
        cmd, text = cmd.lower(), text.strip()

        if cmd in ("quit", "exit"):
            break
        elif cmd == "add" and text:
            vec = model.encode(text).tolist()
            note_id = str(notes.count() + 1)
            notes.add(ids=[note_id], embeddings=[vec], documents=[text])
            print(f"EMBEDDED & STORED: {text} (vector dims: {len(vec)})")
        elif cmd == "query" and text:
            if notes.count() == 0:
                print("No notes stored yet. Use: add <text>")
                continue
            qvec = model.encode(text).tolist()
            print(f"QUERY VECTOR generated (dims: {len(qvec)})")
            res = notes.query(
                query_embeddings=[qvec],
                n_results=min(3, notes.count()),
                include=["documents", "distances", "embeddings"],
            )
            for doc, dist in zip(res["documents"][0], res["distances"][0]):
                print(f"MATCH [score: {dist:.4f}]: {doc}")
            sim = cosine_similarity(qvec, res["embeddings"][0][0])
            print(
                f"TOP RESULT CHECK: manual cosine similarity = {sim:.4f} "
                f"(1 - sim = {1 - sim:.4f}) vs ChromaDB distance = {res['distances'][0][0]:.4f}"
            )
        else:
            print("Usage: add <text> | query <text> | quit")


if __name__ == "__main__":
    main()

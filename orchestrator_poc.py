"""Orchestrator POC: tool calling (step 1) + vector memory (step 2) + history management (step 3)."""
import json
import os
import re
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from llm_client import chat

MAX_TOKENS = 1000  # fake context limit, measured in words
KEEP_LAST = 4
TOP_K = 2
SUMMARY_PROMPT = "Summarize this conversation so far in 2-3 sentences. Preserve key facts and decisions."
DB_PATH = os.path.join(os.path.expanduser("~"), "POC", "chroma_data")
EMBED_MODEL = "all-MiniLM-L6-v2"
FACT_WORDS = {"i", "my", "we", "our"}

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}},
            "required": ["city"],
        },
    },
    {
        "name": "search_notes",
        "description": "Search the user's personal notes for a query string.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search keywords"}},
            "required": ["query"],
        },
    },
]


def get_weather(city):
    return {"city": city, "temp_c": 22, "condition": "Partly cloudy", "humidity_pct": 60}


def search_notes(query):
    return [
        {"id": 1, "title": "Meeting notes", "snippet": f"Discussed roadmap ({query})"},
        {"id": 2, "title": "Grocery list", "snippet": "Milk, eggs, coffee"},
        {"id": 3, "title": "Ideas", "snippet": "Build a tool-calling demo"},
    ]


FUNCTIONS = {"get_weather": get_weather, "search_notes": search_notes}


def count_words(history):
    return sum(len(m["content"].split()) for m in history)


def looks_like_fact(text):
    """Heuristic: the message talks about the user (I / my / we / our)."""
    return bool(FACT_WORDS & set(re.findall(r"[a-z]+", text.lower())))


class Orchestrator:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        db = chromadb.PersistentClient(path=DB_PATH)
        # same collection as vector_memory_poc.py
        self.memory = db.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})
        self.history = []  # only user/assistant text; tool exchanges stay within a turn

    def retrieve(self, text):
        print("--- RETRIEVE ---")
        n = self.memory.count()
        if n == 0:
            print("No memories stored yet.")
            return []
        vec = self.embedder.encode(text).tolist()
        res = self.memory.query(
            query_embeddings=[vec], n_results=min(TOP_K, n), include=["documents", "distances"]
        )
        found = list(zip(res["documents"][0], res["distances"][0]))
        for doc, dist in found:
            print(f"[distance {dist:.4f}] {doc}")
        return [doc for doc, _ in found]

    @staticmethod
    def build_system_prompt(memories):
        body = "\n".join(f"- {m}" for m in memories) if memories else "No relevant memories found"
        return (
            "You are a helpful assistant. Use the memories below when relevant, "
            "and use tools when they help answer the question.\n\n"
            f"RELEVANT MEMORIES:\n{body}"
        )

    def run_tools(self, system, messages):
        """ReAct-style loop: call the model, run any requested tools, repeat until text."""
        response = chat(messages, system, TOOLS)
        while response["type"] == "tool_use":
            print("--- TOOL CALL ---")
            print(f"{response['name']}({json.dumps(response['input'])})")
            try:
                result = json.dumps(FUNCTIONS[response["name"]](**response["input"]))
            except Exception as e:  # report failures back to the model
                result = f"Error: {e}"
            print(f"result: {result}")
            messages.append({"role": "assistant", "tool_call": {
                "id": response["id"], "name": response["name"], "input": response["input"]}})
            messages.append({"role": "tool", "tool_call_id": response["id"],
                             "name": response["name"], "content": result})
            print("--- LLM CALL --- (with tool results)")
            response = chat(messages, system, TOOLS)
        return response["content"]

    def store(self, text):
        print("--- STORE ---")
        if not looks_like_fact(text):
            print("Skipped (no I/my/we/our)")
            return
        self.memory.add(
            ids=[str(uuid.uuid4())],
            embeddings=[self.embedder.encode(text).tolist()],
            documents=[text],
        )
        print(f"Saved memory: {text}")

    def check_history(self):
        if count_words(self.history) > MAX_TOKENS and len(self.history) > KEEP_LAST:
            old, recent = self.history[:-KEEP_LAST], self.history[-KEEP_LAST:]
            summary = chat(
                old + [{"role": "user", "content": SUMMARY_PROMPT}], "You summarize conversations."
            )["content"]
            self.history = [
                {"role": "user", "content": f"[SUMMARY OF EARLIER CONVERSATION]: {summary}"}
            ] + recent
            print(f"CONTEXT LIMIT HIT — summarized {len(old)} old messages")
        print("--- HISTORY ---")
        print(f"{len(self.history)} messages | {count_words(self.history)} words")

    def turn(self, user_text):
        memories = self.retrieve(user_text)
        system = self.build_system_prompt(memories)
        print("--- SYSTEM PROMPT ---")
        print(system[:200] + ("..." if len(system) > 200 else ""))

        self.history.append({"role": "user", "content": user_text})
        print("--- LLM CALL ---")
        print(f"{len(self.history)} messages | {count_words(self.history)} words")
        try:
            reply = self.run_tools(system, list(self.history))
        except Exception as e:
            self.history.pop()  # keep roles alternating
            print(f"API error: {e}")
            return
        self.history.append({"role": "assistant", "content": reply})

        self.store(user_text)
        self.check_history()
        print(f"\nASSISTANT: {reply}")

    def list_memories(self):
        docs = self.memory.get()["documents"]
        print(f"{len(docs)} stored memories:")
        for i, d in enumerate(docs, 1):
            print(f"{i}. {d}")


def main():
    bot = Orchestrator()
    print(f"Loaded {bot.memory.count()} memories. Commands: memories | history | quit")
    while True:
        try:
            user = input("\nUSER: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        cmd = user.lower()
        if cmd in ("quit", "exit"):
            break
        elif cmd == "memories":
            bot.list_memories()
        elif cmd == "history":
            for i, m in enumerate(bot.history):
                print(f"[{i}] {m['role']}: {m['content']}")
        else:
            bot.turn(user)


if __name__ == "__main__":
    main()

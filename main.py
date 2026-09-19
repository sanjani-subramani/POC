"""Unified AI assistant: vector memory + Gemini tool calling + history management + LLM fact extraction."""
import json
import os
import re
import sys
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from llm_client import chat
from orchestrator_poc import FUNCTIONS, TOOLS, count_words  # same tools and word counter as step 4

MAX_WORDS = 1000  # fake context limit, measured in words
KEEP_LAST = 4
TOP_K = 2
DUPLICATE_DISTANCE = 0.05  # facts this close to an existing memory are treated as duplicates
DB_PATH = os.path.join(os.path.expanduser("~"), "POC", "chroma_data")
EMBED_MODEL = "all-MiniLM-L6-v2"
SUMMARY_PROMPT = "Summarize this conversation so far in 2-3 sentences. Preserve key facts and decisions."
EXTRACT_PROMPT = (
    "Extract any personal facts about the user from this message that are worth remembering "
    "long-term. Return ONLY a JSON array of strings. If nothing worth saving, return []. "
    "Message: {message}"
)

FLOW_STEPS = [
    "USER INPUT",
    "[1] RETRIEVE — vector search memories",
    "[2] BUILD PROMPT — inject memories",
    "[3] LLM CALL — send to Gemini",
    "[4] TOOL CHECK — execute if needed, re-call",
    "[5] EXTRACT FACTS — LLM picks what to save",
    "[6] HISTORY CHECK — summarize if too long",
    "RESPONSE",
]


def print_diagram(width=46):
    def row(text):
        return "║" + f" {text}".ljust(width) + "║"

    print("╔" + "═" * width + "╗")
    print(row("AI ASSISTANT PIPELINE"))
    print("╠" + "═" * width + "╣")
    for i, step in enumerate(FLOW_STEPS):
        print(row(step))
        if i < len(FLOW_STEPS) - 1:
            print(row("↓"))
    print("╚" + "═" * width + "╝")


def parse_fact_list(text):
    """Pull a JSON array of strings out of an LLM reply (tolerates code fences and extra prose)."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [f.strip() for f in items if isinstance(f, str) and f.strip()] if isinstance(items, list) else []


class Assistant:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        db = chromadb.PersistentClient(path=DB_PATH)
        self.db = db
        # same collection as vector_memory_poc.py
        self.memory = db.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})
        self.history = []  # user/assistant text only; tool exchanges live inside a single turn

    def embed(self, text):
        return self.embedder.encode(text).tolist()

    # [1] RETRIEVE
    def retrieve(self, text):
        n = self.memory.count()
        if n == 0:
            print("[1/6 RETRIEVE] No memories stored yet.")
            return []
        res = self.memory.query(
            query_embeddings=[self.embed(text)], n_results=min(TOP_K, n),
            include=["documents", "distances"],
        )
        found = list(zip(res["documents"][0], res["distances"][0]))
        print(f"[1/6 RETRIEVE] Found {len(found)} memories:")
        for doc, dist in found:
            print(f"  - [distance {dist:.4f}] {doc}")
        return [doc for doc, _ in found]

    # [2] BUILD PROMPT
    @staticmethod
    def build_prompt(memories):
        body = "\n".join(f"- {m}" for m in memories) if memories else "No relevant memories found"
        system = (
            "You are a helpful assistant. Use the memories below when relevant, "
            "and use tools when they help answer the question.\n\n"
            f"RELEVANT MEMORIES:\n{body}"
        )
        print(f"[2/6 PROMPT] {system[:200]}{'...' if len(system) > 200 else ''}")
        return system

    # [3] LLM CALL + [4] TOOL CHECK
    def respond(self, system):
        messages = list(self.history)  # tool messages are added to this copy only
        print(f"[3/6 LLM] {len(messages)} messages | {count_words(messages)} words")
        response = chat(messages, system, TOOLS)
        used_tool = False
        while response["type"] == "tool_use":
            used_tool = True
            print(f"[4/6 TOOLS] Calling {response['name']}({json.dumps(response['input'])})")
            try:
                result = json.dumps(FUNCTIONS[response["name"]](**response["input"]))
            except Exception as e:  # report failures back to the model
                result = f"Error: {e}"
            print(f"[4/6 TOOLS] Result: {result}")
            messages.append({"role": "assistant", "tool_call": {
                "id": response["id"], "name": response["name"], "input": response["input"]}})
            messages.append({"role": "tool", "tool_call_id": response["id"],
                             "name": response["name"], "content": result})
            print("[3/6 LLM] Re-calling with tool result")
            response = chat(messages, system, TOOLS)
        if not used_tool:
            print("[4/6 TOOLS] No tool needed.")
        return response["content"]

    # [5] EXTRACT FACTS
    def extract_and_store(self, user_text):
        try:
            reply = chat([{"role": "user", "content": EXTRACT_PROMPT.format(message=user_text)}])["content"]
        except Exception as e:
            print(f"[5/6 EXTRACT] Extraction failed ({e}); nothing saved.")
            return
        facts = parse_fact_list(reply)
        if not facts:
            print("[5/6 EXTRACT] No facts worth saving.")
            return
        saved = 0
        for fact in facts:
            vec = self.embed(fact)
            if self.memory.count():
                nearest = self.memory.query(query_embeddings=[vec], n_results=1, include=["distances"])
                if nearest["distances"][0][0] < DUPLICATE_DISTANCE:
                    print(f"[5/6 EXTRACT] Already known, skipped: {fact}")
                    continue
            self.memory.add(ids=[str(uuid.uuid4())], embeddings=[vec], documents=[fact])
            print(f"[5/6 EXTRACT] Saved: {fact}")
            saved += 1
        print(f"[5/6 EXTRACT] {saved} new memories saved.")

    # [6] HISTORY CHECK
    def check_history(self):
        words = count_words(self.history)
        if words > MAX_WORDS and len(self.history) > KEEP_LAST:
            old, recent = self.history[:-KEEP_LAST], self.history[-KEEP_LAST:]
            summary = chat(
                old + [{"role": "user", "content": SUMMARY_PROMPT}], "You summarize conversations."
            )["content"]
            self.history = [
                {"role": "user", "content": f"[SUMMARY OF EARLIER CONVERSATION]: {summary}"}
            ] + recent
            print(f"[6/6 HISTORY] CONTEXT LIMIT HIT — summarized {len(old)} old messages")
        print(f"[6/6 HISTORY] {len(self.history)} messages | {count_words(self.history)} words "
              f"(limit {MAX_WORDS})")

    def turn(self, user_text):
        memories = self.retrieve(user_text)
        system = self.build_prompt(memories)
        self.history.append({"role": "user", "content": user_text})
        try:
            reply = self.respond(system)
        except Exception as e:
            self.history.pop()  # keep roles alternating
            print(f"ERROR: {e}")
            return
        self.history.append({"role": "assistant", "content": reply})
        self.extract_and_store(user_text)
        self.check_history()
        print(f"\nASSISTANT: {reply}")

    def list_memories(self):
        docs = self.memory.get()["documents"]
        print(f"{len(docs)} stored memories:")
        for i, d in enumerate(docs, 1):
            print(f"{i}. {d}")

    def clear_memories(self):
        n = self.memory.count()
        if input(f"Delete all {n} memories? (y/N) ").strip().lower() != "y":
            print("Cancelled.")
            return
        self.db.delete_collection("notes")
        self.memory = self.db.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})
        print(f"Deleted {n} memories.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # box-drawing and arrows on Windows consoles
    print_diagram()
    bot = Assistant()
    print(f"\nLoaded {bot.memory.count()} memories. Commands: memories | history | clear | quit")
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
        elif cmd == "clear":
            bot.clear_memories()
        else:
            bot.turn(user)


if __name__ == "__main__":
    main()

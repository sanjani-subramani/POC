# Tool Calling POC

A minimal command-line demo of LLM **tool calling** (function calling) with Claude, using only the `anthropic` Python SDK.

## What it demonstrates

1. Declaring tools as JSON schemas (`get_weather`, `search_notes`).
2. Sending the user message plus tool definitions to the API.
3. Detecting `tool_use` blocks (`stop_reason == "tool_use"`), running the matching Python function, and returning a `tool_result`.
4. Getting the final natural-language answer, or a direct answer when no tool is needed.

The tools return hardcoded fake data; the point is the loop, not the data.

## Run

```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python tool_calling_poc.py
```

Try: `What's the weather in Paris?`, `Find my notes about roadmap`, or `Hi`. Type `quit` to exit.

Each step is printed: `USER`, `MODEL CHOSE TOOL`, `TOOL RESULT`, `FINAL RESPONSE`.

## Notes

- Model is set by `MODEL` in the script (`claude-sonnet-4-5`).
- Conversation history is kept across turns within a session.

## Step 2: Vector memory (`vector_memory_poc.py`)

Stores short notes as embeddings in a local ChromaDB (`~/POC/chroma_data/`) and retrieves them by meaning. Embeddings come from `all-MiniLM-L6-v2` via `sentence-transformers`, so it runs locally with no API key.

```
pip install -r requirements.txt
python vector_memory_poc.py
> add I love hiking in the mountains
> query outdoor activities
```

Commands: `add <text>`, `query <text>` (top 3 matches with distance scores), `quit`.

### Embeddings
An embedding is a list of numbers (384 for this model) that represents the meaning of a piece of text.
Texts with similar meaning end up as vectors that point in similar directions, even with no shared words.
Because a model turns text into numbers, "similar meaning" becomes a math problem.
Here they are generated locally and stored in ChromaDB.

### Cosine similarity
Cosine similarity measures the angle between two vectors: `dot(a, b) / (|a| * |b|)`.
It ranges from -1 to 1, where 1 means the vectors point the same way (very similar meaning).
ChromaDB reports a distance; with a cosine collection, distance = 1 - similarity, so lower is closer.
The script computes the similarity manually with numpy for the top result so you can compare both numbers.

### RAG (Retrieval-Augmented Generation)
RAG gives an LLM relevant facts at question time instead of relying only on what it memorised in training.
The steps: embed the question, retrieve the most similar stored notes, and paste them into the prompt.
This lets a model answer from your private or up-to-date data without retraining.
This step covers the retrieval half; passing the matches to Claude (as in Step 1) is the generation half.

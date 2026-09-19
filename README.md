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

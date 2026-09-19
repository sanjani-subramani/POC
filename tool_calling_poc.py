"""Minimal tool-calling POC (Gemini): interactive loop with two fake tools."""
import json

from llm_client import chat

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


def run_turn(messages):
    response = chat(messages, tools=TOOLS)
    while response["type"] == "tool_use":
        print(f"MODEL CHOSE TOOL: {response['name']} with args: {json.dumps(response['input'])}")
        messages.append({"role": "assistant", "tool_call": {
            "id": response["id"], "name": response["name"], "input": response["input"]}})
        try:
            result = json.dumps(FUNCTIONS[response["name"]](**response["input"]))
        except Exception as e:  # report failures back to the model
            result = f"Error: {e}"
        print(f"TOOL RESULT: {result}")
        messages.append({"role": "tool", "tool_call_id": response["id"],
                         "name": response["name"], "content": result})
        response = chat(messages, tools=TOOLS)
    messages.append({"role": "assistant", "content": response["content"]})
    return response["content"]


def main():
    messages = []
    while True:
        user = input("\n> ").strip()
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue
        print(f"USER: {user}")
        messages.append({"role": "user", "content": user})
        n = len(messages)
        try:
            print(f"FINAL RESPONSE: {run_turn(messages)}")
        except Exception as e:
            print(f"API ERROR: {e}")
            del messages[n - 1:]  # drop the failed turn, keep roles alternating


if __name__ == "__main__":
    main()

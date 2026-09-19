"""Minimal Claude tool-calling POC: interactive loop with two fake tools."""
import json
import anthropic

MODEL = "claude-sonnet-4-5"  # alias for the latest Sonnet 4.5 snapshot

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


def run_turn(client, messages):
    response = client.messages.create(
        model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
    )
    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"MODEL CHOSE TOOL: {block.name} with args: {json.dumps(block.input)}")
            try:
                result = json.dumps(FUNCTIONS[block.name](**block.input))
                is_error = False
            except Exception as e:  # report failures back to the model
                result, is_error = f"Error: {e}", True
            print(f"TOOL RESULT: {result}")
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_error}
            )
        messages.append({"role": "user", "content": results})
        response = client.messages.create(
            model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
        )
    messages.append({"role": "assistant", "content": response.content})
    return "".join(b.text for b in response.content if b.type == "text")


def main():
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    messages = []
    while True:
        user = input("\n> ").strip()
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue
        print(f"USER: {user}")
        messages.append({"role": "user", "content": user})
        try:
            print(f"FINAL RESPONSE: {run_turn(client, messages)}")
        except anthropic.APIError as e:
            print(f"API ERROR: {e}")
            messages.pop()


if __name__ == "__main__":
    main()

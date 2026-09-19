"""Conversation history management with a fake context window limit.

Keeps full history, counts words after every turn, and when the count exceeds
MAX_TOKENS summarizes all but the last KEEP_LAST messages into one message.
"""
import anthropic

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1000  # fake limit, measured in words
KEEP_LAST = 4
SUMMARY_PROMPT = "Summarize this conversation so far in 2-3 sentences. Preserve key facts and decisions."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


def count_words(history):
    return sum(len(m["content"].split()) for m in history)


def ask(messages):
    response = client.messages.create(model=MODEL, max_tokens=1024, messages=messages)
    return response.content[0].text


def summarize_if_needed(history):
    """Replace all but the last KEEP_LAST messages with one summary message."""
    if count_words(history) <= MAX_TOKENS or len(history) <= KEEP_LAST:
        return history
    old, recent = history[:-KEEP_LAST], history[-KEEP_LAST:]
    summary = ask(old + [{"role": "user", "content": SUMMARY_PROMPT}])
    print(f"CONTEXT LIMIT HIT — summarized {len(old)} old messages")
    return [{"role": "user", "content": f"[SUMMARY OF EARLIER CONVERSATION]: {summary}"}] + recent


def main():
    history = []
    print("Type 'history' to see raw messages, 'quit' to exit.")
    while True:
        print(f"\nHISTORY SIZE: {count_words(history)} words | {len(history)} messages")
        try:
            user_input = input("USER: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "history":
            for i, m in enumerate(history):
                print(f"[{i}] {m['role']}: {m['content']}")
            continue

        history.append({"role": "user", "content": user_input})
        try:
            reply = ask(history)
        except anthropic.APIError as e:
            history.pop()  # keep roles alternating
            print(f"API error: {e}")
            continue
        history.append({"role": "assistant", "content": reply})
        print(f"ASSISTANT: {reply}")

        history = summarize_if_needed(history)


if __name__ == "__main__":
    main()

"""Unified LLM client: Groq first, Gemini as fallback, one normalized interface.

Neutral message format used by the scripts:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str}
  {"role": "assistant", "tool_call": {"id", "name", "input"}}          # model asked for a tool
  {"role": "tool", "tool_call_id", "name", "content": str}             # result of that tool

Tools use a provider-neutral schema: {"name", "description", "input_schema"}.

chat() returns {"type": "text", "content": str}
            or {"type": "tool_use", "name": str, "input": dict, "id": str}
(only the first tool call is returned if the model requests several).
"""
import json
import uuid

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 1024


# ---------- Groq (OpenAI-compatible chat completions) ----------

def _groq_messages(messages, system_prompt):
    out = [{"role": "system", "content": system_prompt}] if system_prompt else []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif "tool_call" in m:
            tc = m["tool_call"]
            out.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])},
                }],
            })
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _chat_groq(messages, system_prompt, tools):
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from env
    kwargs = {}
    if tools:
        kwargs["tools"] = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ]
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=_groq_messages(messages, system_prompt),
        max_tokens=MAX_OUTPUT_TOKENS,
        **kwargs,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        call = msg.tool_calls[0]
        return {
            "type": "tool_use",
            "name": call.function.name,
            "input": json.loads(call.function.arguments or "{}"),
            "id": call.id,
        }
    return {"type": "text", "content": msg.content or ""}


# ---------- Gemini (google-genai) ----------

def _gemini_contents(messages):
    from google.genai import types

    contents = []
    for m in messages:
        if m["role"] == "tool":
            part = types.Part.from_function_response(name=m["name"], response={"result": m["content"]})
            contents.append(types.Content(role="user", parts=[part]))
        elif "tool_call" in m:
            tc = m["tool_call"]
            part = types.Part.from_function_call(name=tc["name"], args=tc["input"])
            contents.append(types.Content(role="model", parts=[part]))
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    return contents


def _chat_gemini(messages, system_prompt, tools):
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from env
    config = {"max_output_tokens": MAX_OUTPUT_TOKENS}
    if system_prompt:
        config["system_instruction"] = system_prompt
    if tools:
        config["tools"] = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"])
            for t in tools
        ])]
        # we run the tool loop ourselves
        config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_gemini_contents(messages),
        config=types.GenerateContentConfig(**config),
    )
    if resp.function_calls:
        call = resp.function_calls[0]
        return {
            "type": "tool_use",
            "name": call.name,
            "input": dict(call.args or {}),
            "id": call.id or f"call_{uuid.uuid4().hex[:8]}",
        }
    return {"type": "text", "content": resp.text or ""}


# ---------- public API ----------

def chat(messages, system_prompt=None, tools=None):
    """Call Groq; on any failure fall back to Gemini. Returns a normalized response dict."""
    try:
        result = _chat_groq(messages, system_prompt, tools)
        print("PROVIDER: Groq")
        return result
    except Exception as groq_error:
        print(f"Groq failed ({type(groq_error).__name__}: {groq_error})")
    result = _chat_gemini(messages, system_prompt, tools)  # raises if Gemini fails too
    print("PROVIDER: Gemini (fallback)")
    return result

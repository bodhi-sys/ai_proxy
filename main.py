import os
import httpx
import json
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")

TOOL_INSTRUCTION = """
You have access to the following tools. To call a tool, respond with a JSON object inside <tool_call> tags.
Format: <tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>

Tools:
%s
"""

def extract_tool_calls(text):
    """
    Extracts tool calls from the model's text response.
    Expects format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            call_data = json.loads(match.strip())
            tool_calls.append({
                "id": f"call_{i}_{os.urandom(4).hex()}",
                "type": "function",
                "function": {
                    "name": call_data.get("name"),
                    "arguments": json.dumps(call_data.get("arguments", {}))
                }
            })
        except json.JSONDecodeError:
            continue
    return tool_calls

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    tools = body.pop("tools", None)
    tool_choice = body.pop("tool_choice", None)

    if tools:
        # Inject tool definitions into the system prompt
        tools_str = json.dumps(tools, indent=2)
        instruction = TOOL_INSTRUCTION % tools_str

        messages = body.get("messages", [])
        system_msg_index = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_msg_index = i
                break

        if system_msg_index != -1:
            messages[system_msg_index]["content"] += "\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})

        body["messages"] = messages

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                UPSTREAM_URL,
                json=body,
                headers=headers,
                timeout=60.0
            )

            if response.status_code != 200:
                return JSONResponse(content=response.json(), status_code=response.status_code)

            resp_json = response.json()

            # If we had tools, try to parse tool calls from the response
            if tools and "choices" in resp_json:
                for choice in resp_json["choices"]:
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    if content:
                        tool_calls = extract_tool_calls(content)
                        if tool_calls:
                            message["tool_calls"] = tool_calls
                            # Optional: remove the raw <tool_call> from content
                            message["content"] = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
                            # If content is empty after stripping, OpenAI usually expects it to be null or an empty string
                            if not message["content"]:
                                message["content"] = None
                            choice["finish_reason"] = "tool_calls"

            return JSONResponse(content=resp_json, status_code=response.status_code)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
import re
import dspy
import litellm
import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure litellm to be more lenient
litellm.drop_params = True

app = FastAPI()

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")

TOOL_INSTRUCTION = """
You have access to the following tools. To call a tool, respond with a JSON object inside <tool_call> tags.
Format: <tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>

Tools:
%s
"""

class ChatCompletionSignature(dspy.Signature):
    """
    You are a helpful assistant. You may have access to tools.
    If tools are available, call them using the specified format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    messages = dspy.InputField(desc="The list of messages in the conversation history.")
    response_content = dspy.OutputField(desc="The assistant's response, potentially including tool calls.")

class ChatProxy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ChatCompletionSignature)

    def forward(self, messages):
        return self.predictor(messages=messages)

def extract_tool_calls(text):
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

async def stream_generator(response_text, model_name):
    chat_id = f"chatcmpl-{os.urandom(12).hex()}"
    created = int(time.time())

    tool_calls = extract_tool_calls(response_text)
    content = response_text

    if tool_calls:
        content = re.sub(r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL).strip()
        if not content:
            content = None

        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": tool_calls
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    if content:
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": content
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    headers = dict(request.headers)

    logger.debug(f"Incoming Request Body: {json.dumps(body, indent=2)}")
    logger.debug(f"Incoming Request Headers: {json.dumps({k: v for k, v in headers.items() if k.lower() != 'authorization'}, indent=2)}")

    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    model_name = body.get("model", "gpt-3.5-turbo")
    messages = body.get("messages", [])
    tools = body.get("tools")
    stream = body.get("stream", False)

    # Inject tools info into system prompt if tools are present
    if tools:
        tools_str = json.dumps(tools, indent=2)
        instruction = TOOL_INSTRUCTION % tools_str

        system_msg_index = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_msg_index = i
                break

        if system_msg_index != -1:
            messages[system_msg_index]["content"] += "\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})

    api_base = UPSTREAM_URL.replace("/chat/completions", "")

    dspy_model_name = model_name
    if not ("/" in dspy_model_name):
        dspy_model_name = f"openai/{dspy_model_name}"

    lm = dspy.LM(model=dspy_model_name, api_key=api_key, api_base=api_base)

    with dspy.context(lm=lm):
        proxy = ChatProxy()

        try:
            # We must ensure that the messages are in the format expected by DSPy.
            # Convert roles to strings or objects that DSPy's adapter understands.
            # Actually, passing the list of dicts should work if the adapter is correctly configured.
            prediction = proxy(messages=messages)
            response_text = prediction.response_content
        except Exception as e:
            logger.error(f"Error in DSPy prediction: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(e, 'lm_response'):
                 response_text = e.lm_response
            else:
                 raise HTTPException(status_code=500, detail=str(e))

        if stream:
            return StreamingResponse(
                stream_generator(response_text, model_name),
                media_type="text/event-stream"
            )

        tool_calls = extract_tool_calls(response_text)

        message = {"role": "assistant", "content": response_text}
        finish_reason = "stop"

        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = re.sub(r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL).strip()
            if not message["content"]:
                message["content"] = None
            finish_reason = "tool_calls"

        return JSONResponse(content={
            "id": f"chatcmpl-{os.urandom(12).hex()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }]
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
import re
import dspy
import litellm
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Configure litellm to be more lenient
litellm.drop_params = True

app = FastAPI()

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")

class ChatCompletionSignature(dspy.Signature):
    """
    You are a helpful assistant. You may have access to tools.
    If tools are available, call them using the specified format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    messages = dspy.InputField(desc="The list of messages in the conversation.")
    tools = dspy.InputField(desc="The list of available tools.")
    response_content = dspy.OutputField(desc="The assistant's response, potentially including tool calls.")

class ChatProxy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ChatCompletionSignature)

    def forward(self, messages, tools=None):
        return self.predictor(messages=messages, tools=tools)

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

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    model_name = body.get("model", "gpt-3.5-turbo")
    messages = body.get("messages", [])
    tools = body.get("tools")

    api_base = UPSTREAM_URL.replace("/chat/completions", "")

    dspy_model_name = model_name
    if not ("/" in dspy_model_name):
        dspy_model_name = f"openai/{dspy_model_name}"

    lm = dspy.LM(model=dspy_model_name, api_key=api_key, api_base=api_base)

    with dspy.context(lm=lm):
        proxy = ChatProxy()

        try:
            prediction = proxy(messages=json.dumps(messages), tools=json.dumps(tools) if tools else "None")
            response_text = prediction.response_content
        except Exception as e:
            if hasattr(e, 'lm_response'):
                 response_text = e.lm_response
            else:
                 raise HTTPException(status_code=500, detail=str(e))

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

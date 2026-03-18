import json
import pytest
from fastapi.testclient import TestClient
from main import app
import respx
import httpx

client = TestClient(app)

@pytest.mark.asyncio
async def test_proxy_without_tools():
    # Setup mock for upstream
    with respx.mock:
        respx.post("https://api.openai.com/v1/chat/completions").mock(return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        }))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hello!"

@pytest.mark.asyncio
async def test_proxy_with_tools_injection():
    with respx.mock as respx_mock:
        def side_effect(request):
            body = json.loads(request.content)
            # Check if instructions were added to system prompt
            messages = body["messages"]
            assert any("You have access to the following tools" in m["content"] for m in messages if m["role"] == "system")
            # Check if tools were stripped
            assert "tools" not in body
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "I will help with that."}}]
            })

        respx_mock.post("https://api.openai.com/v1/chat/completions").side_effect = side_effect

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
            }
        )

        assert response.status_code == 200

@pytest.mark.asyncio
async def test_proxy_with_tool_call_parsing():
    with respx.mock as respx_mock:
        respx_mock.post("https://api.openai.com/v1/chat/completions").mock(return_value=httpx.Response(200, json={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check. <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Francisco\"}}</tool_call>"
                }
            }]
        }))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Weather in SF?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
            }
        )

        assert response.status_code == 200
        data = response.json()
        message = data["choices"][0]["message"]
        assert "tool_calls" in message
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"])["location"] == "San Francisco"
        # The tool call tag should be stripped from content
        assert "Let me check." in message["content"]
        assert "<tool_call>" not in message["content"]

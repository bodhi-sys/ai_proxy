import json
import pytest
from fastapi.testclient import TestClient
from main import app
import respx
import httpx
import dspy

client = TestClient(app)

@pytest.mark.asyncio
async def test_proxy_without_tools():
    with respx.mock as respx_mock:
        respx_mock.post(url__regex=r".*/chat/completions").mock(return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        }))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}]
            },
            headers={"Authorization": "Bearer test-key"}
        )

        assert response.status_code == 200
        assert "Hello!" in response.json()["choices"][0]["message"]["content"]

@pytest.mark.asyncio
async def test_proxy_with_tool_call_parsing():
    with respx.mock as respx_mock:
        respx_mock.post(url__regex=r".*/chat/completions").mock(return_value=httpx.Response(200, json={
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
            },
            headers={"Authorization": "Bearer test-key"}
        )

        assert response.status_code == 200
        data = response.json()
        message = data["choices"][0]["message"]
        assert "tool_calls" in message
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"])["location"] == "San Francisco"
        assert "Let me check." in message["content"]
        assert "<tool_call>" not in message["content"]

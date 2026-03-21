import json
import pytest
from fastapi.testclient import TestClient
from main import app
import unittest.mock as mock
import dspy

client = TestClient(app)

# DSPy expects its LM response choices to have dot access to attributes
class MockChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason

class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

@pytest.fixture
def mock_litellm_completion():
    with mock.patch("litellm.completion") as mocked:
        yield mocked

@pytest.mark.asyncio
async def test_proxy_without_tools(mock_litellm_completion):
    mock_litellm_completion.return_value = {
        "choices": [MockChoice(message=MockMessage(content="Hello!", role="assistant"), finish_reason="stop")],
        "model": "gpt-4"
    }

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
async def test_proxy_with_tool_call_parsing(mock_litellm_completion):
    mock_litellm_completion.return_value = {
        "choices": [MockChoice(message=MockMessage(content="Let me check. <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Francisco\"}}</tool_call>", role="assistant"), finish_reason="stop")],
        "model": "gpt-4"
    }

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

@pytest.mark.asyncio
async def test_proxy_streaming(mock_litellm_completion):
    mock_litellm_completion.return_value = {
        "choices": [MockChoice(message=MockMessage(content="Hello", role="assistant"), finish_reason="stop")],
        "model": "gpt-4"
    }

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        },
        headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.text
    assert "chat.completion.chunk" in content
    assert "Hello" in content
    assert "data: [DONE]" in content

@pytest.mark.asyncio
async def test_proxy_structured_history(mock_litellm_completion):
    def side_effect(*args, **kwargs):
        return {
            "choices": [MockChoice(message=MockMessage(content="The weather is nice.", role="assistant"), finish_reason="stop")],
            "model": "gpt-4"
        }

    mock_litellm_completion.side_effect = side_effect

    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How can I help? <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}</tool_call>"},
        {"role": "user", "content": "The weather is 72 degrees."}
    ]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": history,
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        },
        headers={"Authorization": "Bearer test-key"}
    )

    assert response.status_code == 200
    assert "The weather is nice." in response.json()["choices"][0]["message"]["content"]

    args, kwargs = mock_litellm_completion.call_args
    sent_messages = kwargs.get("messages")
    # Verify that history was preserved correctly and tool instruction was injected
    assert any("You have access to the following tools" in m["content"] for m in sent_messages if m["role"] == "system")
    assert any(m["role"] == "user" and m["content"] == "Hello" for m in sent_messages)
    assert any(m["role"] == "assistant" and "tool_call" in m["content"] for m in sent_messages)
    assert any(m["role"] == "user" and "72 degrees" in m["content"] for m in sent_messages)

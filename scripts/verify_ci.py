import subprocess
import time
import requests
import os
import signal
import sys
import shutil

# Configuration
DUCKAI_DIR = "duckai_repo"
DUCKAI_BIN = os.path.join(DUCKAI_DIR, "target", "release", "duckai")
PROXY_PORT = 8000
DUCKAI_PORT = 8080

def setup_duckai():
    if not os.path.exists(DUCKAI_DIR):
        print(f"Error: {DUCKAI_DIR} not found. In CI, please ensure DuckAI is available.")
        sys.exit(1)

    if not os.path.exists(DUCKAI_BIN):
        print("Building DuckAI (this may take a few minutes)...")
        subprocess.run(["cargo", "build", "--release"], cwd=DUCKAI_DIR, check=True)

def start_process(command, env=None):
    process = subprocess.Popen(
        command,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    return process

def stop_process(process):
    if process:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            pass

def test_chat_completion(model="gpt-5-mini"):
    print(f"Testing chat completion with model {model}...")
    url = f"http://localhost:{PROXY_PORT}/v1/chat/completions"
    headers = {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, who are you?"}]
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    print(f"Chat status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"Received content: {content}")
        return bool(content)
    else:
        print(f"Error: {response.text}")
        return False

def test_tool_call(model="gpt-5-mini"):
    print(f"Testing tool call with model {model}...")
    url = f"http://localhost:{PROXY_PORT}/v1/chat/completions"
    headers = {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    print(f"Tool call status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        message = data["choices"][0]["message"]
        if "tool_calls" in message:
            print(f"Received tool call: {message['tool_calls'][0]['function']['name']}")
            return True
        else:
            print(f"No tool call received. Response: {message.get('content')}")
            return False
    else:
        print(f"Error: {response.text}")
        return False

def main():
    duckai_process = None
    proxy_process = None
    success = False

    try:
        setup_duckai()

        print(f"Starting DuckAI on port {DUCKAI_PORT}...")
        duckai_process = start_process(f"{DUCKAI_BIN} run")
        time.sleep(5)

        print(f"Starting ai_proxy on port {PROXY_PORT}...")
        env = os.environ.copy()
        env["UPSTREAM_URL"] = f"http://localhost:{DUCKAI_PORT}/v1/chat/completions"
        env["PYTHONPATH"] = "."
        proxy_process = start_process(f"{sys.executable} -m uvicorn main:app --port {PROXY_PORT}", env=env)
        time.sleep(10)

        print("Running tests...")
        chat_ok = test_chat_completion()
        tool_ok = test_tool_call()

        success = chat_ok and tool_ok

    except Exception as e:
        print(f"Verification failed: {e}")
    finally:
        print("Cleaning up processes...")
        stop_process(proxy_process)
        stop_process(duckai_process)

    if success:
        print("CI verification PASSED!")
        sys.exit(0)
    else:
        print("CI verification FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()

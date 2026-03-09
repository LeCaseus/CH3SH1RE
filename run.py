import subprocess
import time
import sys

DEVNULL = subprocess.DEVNULL

print("Starting model server...")

model = subprocess.Popen([
    "C:\\AI\\llama.cpp\\llama-server.exe",
    "-m", "C:\\AI\\models\\qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    "--n-gpu-layers", "20",
    "--ctx-size", "2048"
], stdout=DEVNULL, stderr=DEVNULL)

time.sleep(5)

print("Starting vision service...")

vision = subprocess.Popen([
    sys.executable,
    "vision_service.py"
], stdout=DEVNULL, stderr=DEVNULL)

time.sleep(3)

print("Starting assistant controller...")

assistant = subprocess.Popen([
    sys.executable,
    "-m",
    "assistant.main"
], stdout=None, stderr=None)

assistant.wait()
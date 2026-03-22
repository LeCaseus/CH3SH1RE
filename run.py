import subprocess
import sys

print("Starting CH3SH1RE...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
)

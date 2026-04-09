import os

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B")
MODEL_PATH = os.environ.get("MODEL_PATH", MODEL_NAME)  # local path or HF hub ID

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

# Batching: collect requests for up to BATCH_TIMEOUT seconds before dispatching
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
BATCH_TIMEOUT = float(os.environ.get("BATCH_TIMEOUT", "0.05"))  # seconds

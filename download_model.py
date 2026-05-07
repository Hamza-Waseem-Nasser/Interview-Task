import os
from huggingface_hub import snapshot_download

print("Downloading intfloat/multilingual-e5-base...")
try:
    path = snapshot_download(repo_id="intfloat/multilingual-e5-base", local_files_only=False)
    print(f"Downloaded to {path}")
except Exception as e:
    print(f"Error: {e}")

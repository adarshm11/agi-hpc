#!/usr/bin/env python3
"""Download Gemma 4 E4B GGUF for the Ego (CPU inference)."""

import os
from huggingface_hub import hf_hub_download, list_repo_files

repo = "unsloth/gemma-4-E4B-it-GGUF"
dest = "/home/claude/models"
link_name = "gemma-4-E4B-it-Q5_K_M.gguf"

print("Listing files in", repo)
try:
    files = list_repo_files(repo)
except Exception as e:
    print("Repo not found:", e)
    print("Trying alternative names...")
    for alt in [
        "unsloth/gemma-4-4b-it-GGUF",
        "bartowski/gemma-4-E4B-it-GGUF",
        "bartowski/gemma-4-4b-it-GGUF",
    ]:
        try:
            files = list_repo_files(alt)
            repo = alt
            print("Found:", repo)
            break
        except Exception:
            continue
    else:
        print("No repo found. Exiting.")
        exit(1)

gguf_files = [f for f in files if f.endswith(".gguf")]
print("Available GGUFs:")
for f in gguf_files:
    print(" ", f)

# Pick Q5_K_M, fall back to Q4_K_M
target = None
for pattern in ["Q5_K_M", "Q4_K_M"]:
    matches = [f for f in gguf_files if pattern in f]
    if matches:
        target = matches[0]
        break

if not target and gguf_files:
    target = gguf_files[0]

if not target:
    print("No GGUF files found!")
    exit(1)

print()
print("Downloading:", target)
path = hf_hub_download(
    repo_id=repo,
    filename=target,
    local_dir=dest,
)
size_gb = os.path.getsize(path) / 1e9
print("Downloaded to:", path)
print("Size: {:.1f} GB".format(size_gb))

# Symlink with standard name
link_path = os.path.join(dest, link_name)
if not os.path.exists(link_path) and path != link_path:
    os.symlink(path, link_path)
    print("Symlink:", link_path)

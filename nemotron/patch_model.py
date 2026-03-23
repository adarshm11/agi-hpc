"""Patch Nemotron model code to skip fused mamba kernel path (incompatible with 4-bit quantization)."""
import glob
import os

paths = glob.glob(os.path.expanduser("~/.cache/huggingface/**/modeling_nemotron_h.py"), recursive=True)
paths += [os.path.expanduser("~/nemotron/model_cache/modeling_nemotron_h.py")]

for path in paths:
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        content = f.read()

    old = "if self.training and cache_params is None:"
    new = "if False:  # PATCHED: skip fused path for 4-bit quantization"

    if old in content:
        content = content.replace(old, new)
        with open(path, "w") as f:
            f.write(content)
        print(f"PATCHED: {path}")
    elif new in content:
        print(f"ALREADY PATCHED: {path}")
    else:
        print(f"PATTERN NOT FOUND: {path}")

    # Also delete any __pycache__
    cache_dir = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"  Cleared __pycache__: {cache_dir}")

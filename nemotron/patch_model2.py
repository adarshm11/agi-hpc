"""Patch Nemotron model code to fix dtype mismatch in MoE layer."""
import glob
import os

paths = glob.glob(os.path.expanduser("~/.cache/huggingface/**/modeling_nemotron_h.py"), recursive=True)
paths += [os.path.expanduser("~/nemotron/model_cache/modeling_nemotron_h.py")]

for path in paths:
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        content = f.read()

    old = "final_hidden_states.index_add_(0, token_indices, weighted_output)"
    new = "final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))"

    if old in content:
        content = content.replace(old, new)
        with open(path, "w") as f:
            f.write(content)
        print(f"PATCHED dtype fix: {path}")
    elif new in content:
        print(f"ALREADY PATCHED: {path}")
    else:
        print(f"PATTERN NOT FOUND: {path}")

    # Clear pycache
    cache_dir = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)

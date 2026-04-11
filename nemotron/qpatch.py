"""QLoRA patches for Nemotron 3 on Volta GPUs (GV100).

Patches:
  1. Disable caching_allocator_warmup (causes OOM on multi-GPU)
  2. Patch Mamba fused kernel path (incompatible with 4-bit)
  3. Set compute dtype for NF4 quantization
"""

import glob
import os
import shutil


def patch_all(compute_dtype=None):
    """Apply all patches for Nemotron on Volta."""
    _patch_warmup()
    _patch_mamba_fused()
    if compute_dtype is not None:
        _set_compute_dtype(compute_dtype)


def _patch_warmup():
    """Disable transformers caching_allocator_warmup (OOM on GV100)."""
    try:
        import transformers.modeling_utils as mu
        if hasattr(mu, "caching_allocator_warmup"):
            mu.caching_allocator_warmup = lambda *a, **kw: None
            print("[qpatch] Disabled caching_allocator_warmup")
        else:
            print("[qpatch] No caching_allocator_warmup found (OK)")
    except Exception as e:
        print(f"[qpatch] Warning: warmup patch failed: {e}")


def _patch_mamba_fused():
    """Patch Nemotron model to skip fused Mamba kernel (4-bit incompatible)."""
    patterns = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/**/modeling_nemotron_h.py"
        ),
        recursive=True,
    )

    old = "if self.training and cache_params is None:"
    new = "if False:  # PATCHED: skip fused path for 4-bit"

    for path in patterns:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()

        if old in content:
            content = content.replace(old, new)
            with open(path, "w") as f:
                f.write(content)
            # Clear pycache
            cache_dir = os.path.join(os.path.dirname(path), "__pycache__")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            print(f"[qpatch] Patched Mamba fused path: {path}")
        elif new in content:
            print(f"[qpatch] Already patched: {path}")


def _set_compute_dtype(dtype):
    """Set global compute dtype for quantization."""
    import torch
    os.environ["BNB_CUDA_COMPUTE_DTYPE"] = str(dtype)
    print(f"[qpatch] Compute dtype: {dtype}")

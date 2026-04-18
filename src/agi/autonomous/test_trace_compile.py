"""Test trace compiler on all Erebus Python solves."""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")
from compiler.trace_compile import (
    trace_transform, try_constant_output, try_pixel_remap, try_spatial_remap,
)
from grammar.primitives import score_model, verify_model
import onnx

task_dir = Path("/archive/neurogolf")
mem = json.load(open(task_dir / "arc_scientist_memory.json"))

# Get solved transforms
solved = []
for tn_str, tk in mem.get("tasks", {}).items():
    if not tk.get("solved"):
        continue
    for a in reversed(tk.get("attempts", [])):
        if a.get("verified") and a.get("code"):
            solved.append((int(tn_str), a["code"]))
            break

# Existing ONNX
existing = set()
for d in ["solutions_final", "solutions_merged_latest", "solutions_safe", "solutions_conv_v2"]:
    p = task_dir / d
    if p.exists():
        existing.update(int(f.stem[4:]) for f in p.glob("task*.onnx"))

need_onnx = [(tn, code) for tn, code in solved if tn not in existing]
print(f"Erebus solves: {len(solved)}")
print(f"Already have ONNX: {len(solved) - len(need_onnx)}")
print(f"Need ONNX: {len(need_onnx)}")
print()

out_dir = task_dir / "solutions_trace_compiled"
out_dir.mkdir(exist_ok=True)
compiled = 0

for tn, code in sorted(need_onnx):
    tf = task_dir / f"task{tn:03d}.json"
    if not tf.exists():
        continue
    with open(tf) as f:
        task = json.load(f)

    ns = {"np": np, "numpy": np}
    try:
        exec(code.strip(), ns)
        transform_fn = ns.get("transform")
        if not transform_fn:
            continue

        pairs = trace_transform(transform_fn, task)

        result = "no_match"
        for name, fn in [
            ("constant", lambda: try_constant_output(pairs)),
            ("pixel_remap", lambda: try_pixel_remap(pairs)),
            ("spatial_remap", lambda: try_spatial_remap(pairs)),
        ]:
            model = fn()
            if model:
                c, t = verify_model(model, task)
                if c == t and t > 0:
                    s = score_model(model)
                    onnx.save(model, str(out_dir / f"task{tn:03d}.onnx"))
                    result = f"COMPILED ({name}) cost={s['cost']}"
                    compiled += 1
                    break

        # Classify the Python
        inp = np.array(task["train"][0]["input"])
        out = np.array(task["train"][0]["output"])
        shape = f"{inp.shape}->{out.shape}"
        ops = []
        if "for " in code:
            ops.append("loop")
        if "where" in code:
            ops.append("where")
        if "label(" in code or "connected" in code:
            ops.append("cc")
        if "[::-1]" in code or "flip" in code:
            ops.append("flip")

        print(f"task{tn:03d}: {result} | {shape} | {ops}", flush=True)
    except Exception as e:
        print(f"task{tn:03d}: error | {str(e)[:60]}", flush=True)

print(f"\nCompiled: {compiled}/{len(need_onnx)}")

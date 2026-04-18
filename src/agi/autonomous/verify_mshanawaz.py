"""Verify mshanawaz's 400 ONNX files and copy verified ones."""
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, "src")
from grammar.primitives import score_model, verify_model
import onnx

src = Path("/tmp/mshanawaz-ng/onnx_safe_full")
task_dir = Path("/archive/neurogolf")
out = Path("/archive/neurogolf/solutions_mshanawaz_v2")
out.mkdir(exist_ok=True)

verified = failed = 0
total_score = 0

for f in sorted(src.glob("task*.onnx")):
    tn = int(f.stem[4:])
    tf = task_dir / f"task{tn:03d}.json"
    if not tf.exists():
        continue
    try:
        model = onnx.load(str(f))
        with open(tf) as jf:
            task = json.load(jf)
        c, t = verify_model(model, task)
        if c == t and t > 0:
            s = score_model(model)
            cost = s["cost"] if s else 0
            score = s["score"] if s else 1
            shutil.copy2(str(f), str(out / f.name))
            verified += 1
            total_score += score
        else:
            failed += 1
    except Exception:
        failed += 1

    if (verified + failed) % 50 == 0:
        print(f"Progress: {verified} verified, {failed} failed", flush=True)

print(f"\nDone: {verified} verified, {failed} failed")
print(f"Total projected score: {total_score:.0f}")
print(f"Files saved to: {out}")

"""Quick merge — combine all ONNX sources, pick cheapest, build submission."""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "src")

task_dir = sys.argv[1] if len(sys.argv) > 1 else "/archive/neurogolf"

from grammar.primitives import score_model
import onnx

sources = [
    "solutions_safe",
    "solutions_merged_latest",
    "solutions_conv_v2",
    "solutions_parallel",
    "solutions_dagastar",
    "solutions_gpu_atlas",
    "solutions_dagastar_nrp",
    "solutions_erebus_onnx",
]

candidates = {}
for d in sources:
    p = Path(task_dir) / d
    if not p.exists():
        continue
    for f in p.glob("task*.onnx"):
        tn = int(f.stem[4:])
        try:
            model = onnx.load(str(f))
            s = score_model(model)
            if s:
                candidates.setdefault(tn, []).append((s["cost"], s["score"], str(f)))
        except Exception:
            pass

out = Path(task_dir) / "solutions_final"
out.mkdir(exist_ok=True)

total_score = 0
merged = 0
for tn in sorted(candidates.keys()):
    entries = sorted(candidates[tn])
    cost, score, path = entries[0]
    shutil.copy2(path, str(out / f"task{tn:03d}.onnx"))
    total_score += score
    merged += 1

print(f"Merged: {merged} tasks")
print(f"Projected score: {total_score:.0f}")
print(f"Estimated actual (x0.72): {total_score * 0.72:.0f}")

# Build submission.zip
import zipfile

zip_path = Path(task_dir) / "submission.zip"
with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(out.glob("task*.onnx")):
        zf.write(f, f.name)
print(f"Built: {zip_path} ({merged} tasks)")

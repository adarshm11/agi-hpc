#!/usr/bin/env python3
"""Nemotron v3 training on Atlas (2x GV100 32GB).

Adapts nemotron_v3_kaggle.py for local hardware:
  - 4-bit NF4 quantization (GV100 has no bf16)
  - fp16 compute dtype
  - ThermalController from batch-probe
  - LoRA targets: up_proj + down_proj only (Mamba breaks with 4-bit in_proj)
  - Reads data from /home/claude/nemotron/data/

Usage:
    python train_nemotron_v3_atlas.py
    python train_nemotron_v3_atlas.py --skip-sweep  # Skip HP proxy sweep
    python train_nemotron_v3_atlas.py --val-only     # Validation only (no training)
"""

import os
import sys
import time
import random
import re
import math
import argparse
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Apply QLoRA patches (warmup OOM fix + Mamba fused kernel fix)
sys.path.insert(0, "/home/claude/nemotron")
import qpatch
qpatch.patch_all(compute_dtype="float16")

# ── Thermal protection ──────────────────────────────────────
try:
    from batch_probe import ThermalController
    tc = ThermalController(target_temp=82.0, max_threads=20, min_threads=4)
    tc.start()
    print("ThermalController active (target 82C)")
except ImportError:
    tc = None
    print("WARNING: batch-probe not available, no thermal protection")

# ── Parse args ──────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--skip-sweep", action="store_true",
                    help="Skip HP proxy sweep, use default LR")
parser.add_argument("--val-only", action="store_true",
                    help="Load base model and run validation only")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=1.5e-4)
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--data-dir", default="/home/claude/nemotron/data")
args = parser.parse_args()

# ── Import heavy libs ───────────────────────────────────────

import torch
import polars as pl
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__}, Device: {DEVICE}")
if DEVICE == "cuda":
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

# ── Load data ───────────────────────────────────────────────

train_df = pl.read_csv(f"{args.data_dir}/train.csv")
test_df = pl.read_csv(f"{args.data_dir}/test.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ── Import all augmentation + CoT from v3 ──────────────────
# (inline the functions rather than importing — keeps it self-contained)

# Can't import from nemotron_v3_kaggle.py directly — it has top-level
# Kaggle-specific code. Instead, define the shared functions here.
# These are identical to v3's implementations.

def classify_task(prompt):
    p = prompt.lower()
    if "bit manipulation" in p or ("01" in prompt and "->" in prompt):
        return "bit"
    elif "encryption" in p or "decrypt" in p or "cipher" in p:
        return "encrypt"
    elif "gravitational" in p or "g =" in p or "free-fall" in p:
        return "physics"
    elif "unit conversion" in p or "convert" in p:
        return "unit"
    elif "numeral system" in p or "roman" in p:
        return "numeral"
    elif "transformation rules" in p or "symbol" in p:
        return "symbol"
    return "unknown"

def int_to_roman(n):
    vals = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),
            (90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
    result = []
    for val, sym in vals:
        while n >= val:
            result.append(sym)
            n -= val
    return "".join(result)

def roman_to_int(s):
    vals = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    result, prev = 0, 0
    for c in reversed(s.upper()):
        v = vals.get(c, 0)
        result += -v if v < prev else v
        prev = v
    return result

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""

def check_match(predicted, actual):
    if predicted.strip() == actual.strip():
        return True
    try:
        p, a = float(predicted), float(actual)
        return abs(p - a) / max(abs(a), 1e-10) < 0.01
    except (ValueError, ZeroDivisionError):
        return False

def classify_error(predicted, actual, task_type):
    if not predicted: return "empty_output"
    if check_match(predicted, actual): return "correct"
    if "boxed" in predicted or "\\" in predicted: return "format_error"
    try:
        p, a = float(predicted), float(actual)
        if abs(p - a) <= 1: return "off_by_one"
        if abs(p - a) / max(abs(a), 1e-10) < 0.05: return "rounding_error"
    except ValueError: pass
    if task_type in ("encrypt", "symbol") and len(actual) > 0 and len(predicted) > 0:
        common = sum(1 for a, b in zip(actual, predicted) if a == b)
        if common / max(len(actual), len(predicted)) > 0.8: return "partial_match"
    if task_type == "bit" and len(predicted) == 8 and len(actual) == 8:
        diff = sum(1 for a, b in zip(actual, predicted) if a != b)
        if diff <= 2: return f"bit_error_{diff}"
    return "wrong_answer"

# Load augmentation functions from v3 via runpy (import only defs)
# We exec just the function/dict definitions, skipping Kaggle I/O
_v3_source = open("/home/claude/nemotron/nemotron_v3_kaggle.py").read()

# Extract everything between STEP 2 and STEP 3 headers (augmentation code)
_aug_start = _v3_source.find("def classify_task")
_aug_end = _v3_source.find("# ═══════════════════════════════════════════════════════════════\n# STEP 3")
if _aug_start > 0 and _aug_end > _aug_start:
    # Already defined classify_task above, skip re-defining it
    _aug_code = _v3_source[_aug_start:_aug_end]
    # Remove classify_task since we already have it
    _aug_code = _aug_code[_aug_code.find("\n\n") + 2:]
    exec(_aug_code, globals())
    print("  Loaded augmentation functions from v3")

# Extract COT_TEMPLATES
_cot_start = _v3_source.find("COT_TEMPLATES = {")
_cot_end = _v3_source.find("\n}\n", _cot_start) + 2
if _cot_start > 0:
    exec(_v3_source[_cot_start:_cot_end], globals())
    print(f"  Loaded COT_TEMPLATES ({len(COT_TEMPLATES)} types)")

AUGMENT_FNS = {
    "bit": (augment_bit_task, 8),
    "encrypt": (augment_encrypt_task, 5),
    "physics": (augment_physics_task, 5),
    "unit": (augment_unit_task, 5),
    "numeral": (augment_numeral_task, 3),
    "symbol": (augment_symbol_task, 4),
}

# ── Build augmented dataset ─────────────────────────────────

random.seed(42)
tasks = []
for row in train_df.iter_rows(named=True):
    tasks.append({
        "id": row["id"],
        "prompt": row["prompt"],
        "answer": str(row["answer"]),
        "type": classify_task(row["prompt"]),
    })

type_counts = Counter(t["type"] for t in tasks)
print(f"Task types: {dict(type_counts)}")

augmented_tasks = []
aug_counts = {}

for task in tasks:
    augmented_tasks.append(task)
    ttype = task["type"]
    if ttype in AUGMENT_FNS:
        fn, n = AUGMENT_FNS[ttype]
        try:
            augs = fn(task["prompt"], task["answer"], n)
            for new_prompt, new_answer in augs:
                augmented_tasks.append({
                    "id": f"{task['id']}_aug",
                    "prompt": new_prompt,
                    "answer": new_answer,
                    "type": ttype,
                })
                aug_counts[ttype] = aug_counts.get(ttype, 0) + 1
        except Exception:
            pass

print(f"Original: {len(tasks)}, Augmented: {len(augmented_tasks)} "
      f"({len(augmented_tasks)/len(tasks):.1f}x)")
for t in sorted(aug_counts.keys()):
    print(f"  {t}: +{aug_counts[t]}")

random.shuffle(augmented_tasks)


def format_training(task):
    cot = COT_TEMPLATES.get(task["type"],
                            "Let me think through this step by step.\n\n")
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{task['prompt']}\n\n"
        f"Think step by step. Put your final answer in \\boxed{{}}.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{cot}"
        f"\\boxed{{{task['answer']}}}\n"
        f"<|eot_id|>"
    )


formatted = [format_training(t) for t in augmented_tasks]
split_idx = int(len(formatted) * 0.9)
train_texts = formatted[:split_idx]
val_texts = formatted[split_idx:]
val_tasks = augmented_tasks[split_idx:]
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

# ── Load model (4-bit for GV100) ────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Use NVIDIA FP8 pre-quantized model — already small, no bf16→4bit OOM
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"

print(f"\nLoading {MODEL_NAME}...")
print("Strategy: load to CPU first, then quantize to 4-bit on GPU")
t0 = time.time()

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Key fix: load weights to CPU first with offload_folder, then quantize
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    max_memory={0: "15GiB", 1: "15GiB", "cpu": "150GiB"},
    offload_folder="/tmp/nemotron_offload",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded in {time.time()-t0:.0f}s")

# ── Validation-only mode ────────────────────────────────────

if args.val_only:
    print("\n=== VALIDATION ONLY (base model, no LoRA) ===")
    base_model.eval()
    type_correct = {}
    type_total = {}
    total = 0
    correct = 0

    val_originals = [
        t for t in val_tasks if "_aug" not in str(t.get("id", ""))
    ][:30]

    for task in val_originals:
        prompt_text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{task['prompt']}\n\n"
            f"Think step by step. Put your final answer in \\boxed{{}}.\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs, max_new_tokens=512,
                temperature=1.0, do_sample=False
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        pred = extract_boxed(response)
        actual = task["answer"]
        ttype = task.get("type", "unknown")

        match = check_match(pred, actual)
        total += 1
        type_total[ttype] = type_total.get(ttype, 0) + 1
        if match:
            correct += 1
            type_correct[ttype] = type_correct.get(ttype, 0) + 1
        elif total <= 10:
            err = classify_error(pred, actual, ttype)
            print(f"  MISS [{ttype}] {err}: "
                  f"pred='{pred[:30]}' actual='{actual[:30]}'")

    print(f"\nBase model accuracy: {correct}/{total} ({correct/total:.1%})")
    for tt in sorted(type_total.keys()):
        tc = type_correct.get(tt, 0)
        print(f"  {tt:12s}: {tc}/{type_total[tt]} "
              f"({tc/max(type_total[tt],1):.0%})")

    if tc:
        tc.stop()
    sys.exit(0)

# ── LoRA setup ──────────────────────────────────────────────

LORA_RANK = args.rank
best_lr = args.lr

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    # GV100 4-bit: only MLP projections (Mamba in_proj breaks)
    target_modules=["up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── HP proxy sweep (unless skipped) ─────────────────────────

from trl import SFTTrainer, SFTConfig

if not args.skip_sweep:
    CONFIGS = [
        {"lr": 1e-4, "name": "low_lr"},
        {"lr": 1.5e-4, "name": "mid_lr"},
        {"lr": 2e-4, "name": "high_lr"},
    ]

    proxy_size = len(train_texts) // 5
    proxy_train = train_texts[:proxy_size]
    proxy_val = val_texts[:max(len(val_texts) // 3, 50)]

    print(f"\n=== HP Proxy Sweep ({len(CONFIGS)} configs, "
          f"{proxy_size} samples) ===")
    best_loss = float("inf")

    for cfg in CONFIGS:
        print(f"\n--- {cfg['name']} (LR={cfg['lr']}) ---")

        model = get_peft_model(base_model, lora_config)

        proxy_args = SFTConfig(
            output_dir=f"/tmp/nemotron_proxy_{cfg['name']}",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=cfg["lr"],
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=True,  # GV100
            gradient_checkpointing=True,
            max_seq_length=2048,
            dataset_text_field="text",
            report_to="none",
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            args=proxy_args,
            train_dataset=Dataset.from_dict({"text": proxy_train}),
            eval_dataset=Dataset.from_dict({"text": proxy_val}),
            tokenizer=tokenizer,
        )

        result = trainer.train()
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss", float("inf"))
        print(f"  Train: {result.training_loss:.4f}, "
              f"Eval: {eval_loss:.4f}")

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_lr = cfg["lr"]

        # Reset for next config
        del model, trainer
        torch.cuda.empty_cache()
        model = None

    print(f"\n=== Best: LR={best_lr} (eval_loss={best_loss:.4f}) ===")

# ── Full training ───────────────────────────────────────────

print(f"\n=== Full Training (LR={best_lr}, rank={LORA_RANK}, "
      f"epochs={args.epochs}) ===")

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

n_train = len(train_texts)
grad_accum = 8
steps_per_epoch = n_train // (2 * grad_accum)  # batch=2
total_steps = args.epochs * steps_per_epoch
eval_steps = max(total_steps // 8, 50)

training_args = SFTConfig(
    output_dir="/home/claude/nemotron/checkpoints",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=grad_accum,
    learning_rate=best_lr,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,  # GV100
    gradient_checkpointing=True,
    max_seq_length=2048,
    dataset_text_field="text",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=Dataset.from_dict({"text": train_texts}),
    eval_dataset=Dataset.from_dict({"text": val_texts}),
    tokenizer=tokenizer,
)

print("Starting training...")
t0 = time.time()
result = trainer.train()
elapsed = (time.time() - t0) / 60
print(f"Training complete in {elapsed:.1f} min")
print(f"Final loss: {result.training_loss:.4f}")

# ── Validation with error analysis ──────────────────────────

print("\n=== Validation ===")
model.eval()
type_correct = {}
type_total = {}
type_errors = {}
correct = 0
total = 0

val_originals = [
    t for t in val_tasks if "_aug" not in str(t.get("id", ""))
][:50]

for task in val_originals:
    prompt_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{task['prompt']}\n\n"
        f"Think step by step. Put your final answer in \\boxed{{}}.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512,
            temperature=1.0, do_sample=False
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    pred = extract_boxed(response)
    actual = task["answer"]
    ttype = task.get("type", "unknown")

    match = check_match(pred, actual)
    total += 1
    type_total[ttype] = type_total.get(ttype, 0) + 1
    if match:
        correct += 1
        type_correct[ttype] = type_correct.get(ttype, 0) + 1
    else:
        if ttype not in type_errors:
            type_errors[ttype] = []
        type_errors[ttype].append((pred, actual, task["prompt"][:60]))

    if total % 10 == 0:
        print(f"  [{total}/50] accuracy: {correct/total:.0%}")

print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")
print("\nPer-type:")
for tt in sorted(type_total.keys()):
    tc = type_correct.get(tt, 0)
    print(f"  {tt:12s}: {tc}/{type_total[tt]} "
          f"({tc/max(type_total[tt],1):.0%})")

print("\nError analysis:")
for tt in sorted(type_errors.keys()):
    errors = type_errors[tt]
    modes = Counter()
    for pred, actual, _ in errors:
        modes[classify_error(pred, actual, tt)] += 1
    print(f"  {tt} ({len(errors)} errors):")
    for mode, count in modes.most_common():
        print(f"    {mode}: {count}")

# ── Save adapter ────────────────────────────────────────────

print("\nSaving adapter to /home/claude/nemotron/adapter_v3/")
model.save_pretrained("/home/claude/nemotron/adapter_v3")
tokenizer.save_pretrained("/home/claude/nemotron/adapter_v3")
print("Done!")

# ── Cleanup ─────────────────────────────────────────────────

if tc:
    tc.stop()

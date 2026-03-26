"""Nemotron 3 Reasoning Challenge — Kaggle Submission
NVIDIA Competition | Blackwell GPU (RTX PRO 6000, 96GB)

Follows the OFFICIAL demo pattern exactly:
  1. kagglehub.model_download() for model path
  2. import mamba_ssm BEFORE model load (required for Mamba architecture)
  3. bf16, NO quantization (96GB VRAM is plenty for 30B model)
  4. LoRA rank 32, target in_proj/out_proj/up_proj/down_proj
  5. Save adapter -> zip -> submit

Training additions:
  - Group-theoretic data augmentation (Bond, 2026a, Ch. 13)
  - Task-specific chain-of-thought templates
  - Symmetry-consistent augmentation of input-output pairs

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
Enable Blackwell GPU. Add competition data as input.
"""

import os, json, time, random, re, math
import polars as pl

# ═══════════════════════════════════════════════════════════════
# STEP 1: Load competition data
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/nvidia-nemotron-3-reasoning-challenge"
if not os.path.exists(DATA_DIR):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            DATA_DIR = root
            break

train_df = pl.read_csv(f"{DATA_DIR}/train.csv")
test_df = pl.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")


# ═══════════════════════════════════════════════════════════════
# STEP 2: Task classification + group-theoretic augmentation
# ═══════════════════════════════════════════════════════════════

random.seed(42)

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


# --- Bit manipulation augmentation (S_8 x Z_2) ---
def augment_bit_task(prompt, answer, n_augments=3):
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    # Parse examples
    examples = []
    query = None
    for line in prompt.split("\n"):
        line = line.strip()
        if "->" in line and len(line) < 30:
            parts = line.split("->")
            if len(parts) == 2:
                inp, out = parts[0].strip(), parts[1].strip()
                if len(inp) == 8 and len(out) == 8 and all(c in "01" for c in inp + out):
                    examples.append((inp, out))
    for line in reversed(prompt.split("\n")):
        match = re.search(r'([01]{8})', line.strip())
        if match:
            query = match.group(1)
            break

    if not examples or not query:
        return []

    for _ in range(n_augments):
        perm = list(range(8))
        rng.shuffle(perm)
        do_complement = rng.random() < 0.3

        def transform(bits):
            result = "".join(bits[p] for p in perm)
            if do_complement:
                result = "".join("1" if b == "0" else "0" for b in result)
            return result

        new_examples = [(transform(i), transform(o)) for i, o in examples]
        new_query = transform(query)
        new_answer = transform(answer) if len(answer) == 8 and all(c in "01" for c in answer) else answer

        ex_text = "\n".join(f"{i} -> {o}" for i, o in new_examples)
        header = prompt.split("\n")[0] if "\n" in prompt else "Apply the transformation:"
        new_prompt = f"{header}\n{ex_text}\n\nDetermine the output for: {new_query}"
        augmented.append((new_prompt, new_answer))

    return augmented


# --- Encryption augmentation (S_26 relabeling) ---
def augment_encrypt_task(prompt, answer, n_augments=2):
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)
    letters = list("abcdefghijklmnopqrstuvwxyz")

    for _ in range(n_augments):
        perm = list(letters)
        rng.shuffle(perm)
        fwd = {a: b for a, b in zip(letters, perm)}
        fwd.update({a.upper(): b.upper() for a, b in zip(letters, perm)})

        new_prompt = "".join(fwd.get(c, c) for c in prompt)
        new_answer = "".join(fwd.get(c, c) for c in answer)
        augmented.append((new_prompt, new_answer))

    return augmented


# --- Physics augmentation (scale invariance) ---
def augment_physics_task(prompt, answer, n_augments=2):
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    numbers = re.findall(r'(\d+\.?\d*)', prompt)
    if len(numbers) < 3:
        return []

    for _ in range(n_augments):
        scale = rng.uniform(0.5, 2.0)
        new_prompt = prompt
        new_answer = answer
        try:
            new_answer = str(round(float(answer) * scale, 2))
            for num in numbers:
                new_prompt = new_prompt.replace(num, str(round(float(num) * scale, 2)), 1)
        except ValueError:
            continue
        augmented.append((new_prompt, new_answer))

    return augmented


# --- Symbol augmentation (permutation of symbol set) ---
def augment_symbol_task(prompt, answer, n_augments=2):
    symbols = sorted(set(c for c in prompt + answer if c not in " =\n\t" and not c.isalnum()))
    if len(symbols) < 3:
        return []

    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    for _ in range(n_augments):
        perm = list(symbols)
        rng.shuffle(perm)
        mapping = dict(zip(symbols, perm))

        def remap(text):
            result = list(text)
            temp = {}
            for i, c in enumerate(result):
                if c in mapping:
                    temp[i] = mapping[c]
            for i, c in temp.items():
                result[i] = c
            return "".join(result)

        augmented.append((remap(prompt), remap(answer)))

    return augmented


# Apply augmentation
AUGMENT_FNS = {
    "bit": (augment_bit_task, 3),
    "encrypt": (augment_encrypt_task, 2),
    "physics": (augment_physics_task, 2),
    "symbol": (augment_symbol_task, 2),
}

tasks = []
for row in train_df.iter_rows(named=True):
    tasks.append({
        "id": row["id"],
        "prompt": row["prompt"],
        "answer": str(row["answer"]),
        "type": classify_task(row["prompt"]),
    })

type_counts = {}
for t in tasks:
    type_counts[t["type"]] = type_counts.get(t["type"], 0) + 1
print(f"Task types: {type_counts}")

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
                    "id": task["id"] + "_aug",
                    "prompt": new_prompt,
                    "answer": new_answer,
                    "type": ttype,
                })
                aug_counts[ttype] = aug_counts.get(ttype, 0) + 1
        except Exception:
            pass

print(f"Original: {len(tasks)}, After augmentation: {len(augmented_tasks)} "
      f"({len(augmented_tasks)/len(tasks):.1f}x)")
for t in sorted(aug_counts.keys()):
    print(f"  {t}: +{aug_counts[t]} augmented")

random.shuffle(augmented_tasks)


# ═══════════════════════════════════════════════════════════════
# STEP 3: Format training data with chain-of-thought
# ═══════════════════════════════════════════════════════════════

COT_TEMPLATES = {
    "bit": "Let me analyze the bit transformation pattern.\nI'll examine each bit position to find the rule.\n\n",
    "encrypt": "Let me decode the substitution cipher.\nI'll map each cipher character to its plaintext equivalent.\n\n",
    "physics": "I need to find g from the data using d = 0.5*g*t^2.\nFrom the examples: g = 2*d/t^2.\n\n",
    "unit": "I need to find the conversion factor from the examples.\nLet me compute output/input for each pair.\n\n",
    "numeral": "I need to identify the numeral system and apply it.\n\n",
    "symbol": "I need to find the symbol transformation rules.\nLet me map each input symbol to its output.\n\n",
}


def format_training(task):
    cot = COT_TEMPLATES.get(task["type"], "Let me think through this step by step.\n\n")
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
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Load model — EXACTLY following the official demo
# ═══════════════════════════════════════════════════════════════

import kagglehub
import mamba_ssm  # REQUIRED before model load — Nemotron uses Mamba layers
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

OUTPUT_DIR = "/kaggle/working"
LORA_RANK = 32

# Download model via kagglehub (official method)
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)

print(f"Loading model from {MODEL_PATH}...")
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,  # bf16, NO quantization — Blackwell has 96GB
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Model loaded in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════
# STEP 5: Configure LoRA (official demo pattern)
# ═══════════════════════════════════════════════════════════════

print(f"Initializing LoRA adapter with rank={LORA_RANK}...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ═══════════════════════════════════════════════════════════════
# STEP 6: Train
# ═══════════════════════════════════════════════════════════════

from trl import SFTTrainer, SFTConfig

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

n_train = len(train_texts)
n_epochs = 2
grad_accum = 16
steps_per_epoch = n_train // grad_accum
total_steps = n_epochs * steps_per_epoch
eval_steps = max(total_steps // 8, 50)

print(f"\nTraining config:")
print(f"  Dataset: {n_train} examples (augmented)")
print(f"  Epochs: {n_epochs}")
print(f"  Effective batch size: {grad_accum}")
print(f"  Total steps: ~{total_steps}")
print(f"  Eval every: {eval_steps} steps")

training_args = SFTConfig(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=n_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=grad_accum,
    learning_rate=1.5e-4,
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
    bf16=True,  # Blackwell supports bf16
    gradient_checkpointing=True,
    max_seq_length=2048,
    dataset_text_field="text",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("\nStarting training...")
t0 = time.time()
result = trainer.train()
elapsed = (time.time() - t0) / 60
print(f"Training complete in {elapsed:.1f} min")
print(f"Final training loss: {result.training_loss:.4f}")


# ═══════════════════════════════════════════════════════════════
# STEP 7: Quick validation
# ═══════════════════════════════════════════════════════════════

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


print("\nValidating...")
model.eval()
correct = 0
total = 0

val_originals = [t for t in augmented_tasks[split_idx:] if "_aug" not in str(t["id"])][:50]

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
        outputs = model.generate(**inputs, max_new_tokens=512,
                                 temperature=0.0, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    predicted = extract_boxed(response)
    actual = task["answer"]

    match = predicted.strip() == actual.strip()
    if not match:
        try:
            match = abs(float(predicted) - float(actual)) / max(abs(float(actual)), 1e-10) < 0.01
        except (ValueError, ZeroDivisionError):
            pass

    total += 1
    if match:
        correct += 1
    elif total <= 5:
        print(f"  MISS [{task['type']}]: pred='{predicted[:40]}' actual='{actual[:40]}'")
    if total % 10 == 0:
        print(f"  [{total}/50] accuracy: {correct/total:.0%}")

print(f"\nValidation accuracy: {correct}/{total} ({correct/total:.1%})")


# ═══════════════════════════════════════════════════════════════
# STEP 8: Save adapter and package submission
# ═══════════════════════════════════════════════════════════════

print("\nSaving adapter...")
model.save_pretrained(OUTPUT_DIR)

# Verify and zip — EXACTLY like the demo
import subprocess
subprocess.run("zip -m submission.zip *", shell=True, check=True, cwd=OUTPUT_DIR)

print("\n" + "=" * 70)
print("SUBMISSION READY: /kaggle/working/submission.zip")
print("=" * 70)
print("Done.")

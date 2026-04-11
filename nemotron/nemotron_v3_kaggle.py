"""Nemotron 3 Reasoning Challenge v3 — Kaggle Submission
NVIDIA Competition | Blackwell GPU (RTX PRO 6000, 96GB) | $106K

v3 improvements over v2:
  1. Per-type error analysis with failure mode classification
  2. Richer CoT templates with worked reasoning demonstrations
  3. Expanded group-theoretic augmentation (~5x dataset, was ~2x)
  4. Hyperparameter proxy sweep (3 configs on 20% data)
  5. Self-consistency voting for reliable validation
  6. Roman numeral + unit conversion augmentation helpers

Follows the OFFICIAL demo pattern exactly:
  kagglehub.model_download() → import mamba_ssm → bf16 → LoRA → zip

Reference: "Geometric Methods in Computational Modeling" (Bond, 2026a)
  Ch. 13: Group-Theoretic Data Augmentation

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
Enable Blackwell GPU. Add competition data as input.
"""

import os, json, time, random, re, math
from collections import Counter
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


# ── Helpers ───────────────────────────────────────────────────

def int_to_roman(n):
    """Convert integer to Roman numeral string."""
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    result = []
    for val, sym in vals:
        while n >= val:
            result.append(sym)
            n -= val
    return "".join(result)


def roman_to_int(s):
    """Convert Roman numeral string to integer."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50,
            "C": 100, "D": 500, "M": 1000}
    result = 0
    prev = 0
    for c in reversed(s.upper()):
        v = vals.get(c, 0)
        if v < prev:
            result -= v
        else:
            result += v
        prev = v
    return result


# ── Bit manipulation augmentation (S_8 x Z_2 x Z_8) ─────────

def augment_bit_task(prompt, answer, n_augments=8):
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    examples = []
    query = None
    for line in prompt.split("\n"):
        line = line.strip()
        if "->" in line and len(line) < 30:
            parts = line.split("->")
            if len(parts) == 2:
                inp, out = parts[0].strip(), parts[1].strip()
                if (len(inp) == 8 and len(out) == 8
                        and all(c in "01" for c in inp + out)):
                    examples.append((inp, out))
    for line in reversed(prompt.split("\n")):
        match = re.search(r'([01]{8})', line.strip())
        if match:
            query = match.group(1)
            break

    if not examples or not query:
        return []

    header = prompt.split("\n")[0] if "\n" in prompt else "Apply the transformation:"

    for i in range(n_augments):
        if i < 7:
            # Circular shift by i+1 positions (Z_8 subgroup)
            shift = (i + 1) % 8

            def transform(bits, s=shift):
                return bits[s:] + bits[:s]
        else:
            # Random S_8 permutation + optional complement
            perm = list(range(8))
            rng.shuffle(perm)
            do_complement = rng.random() < 0.3

            def transform(bits, p=perm, c=do_complement):
                result = "".join(bits[j] for j in p)
                if c:
                    result = "".join("1" if b == "0" else "0" for b in result)
                return result

        new_examples = [(transform(inp), transform(out)) for inp, out in examples]
        new_query = transform(query)
        new_answer = (
            transform(answer)
            if len(answer) == 8 and all(c in "01" for c in answer)
            else answer
        )

        ex_text = "\n".join(f"{inp} -> {out}" for inp, out in new_examples)
        new_prompt = f"{header}\n{ex_text}\n\nDetermine the output for: {new_query}"
        augmented.append((new_prompt, new_answer))

    return augmented


# ── Encryption augmentation (S_26 relabeling) ────────────────

def augment_encrypt_task(prompt, answer, n_augments=5):
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


# ── Physics augmentation (scale + time + unit invariance) ────

def augment_physics_task(prompt, answer, n_augments=5):
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    numbers = re.findall(r'(\d+\.?\d*)', prompt)
    if len(numbers) < 3:
        return []

    for i in range(n_augments):
        try:
            if i < 2:
                # Scale invariance (original method)
                scale = rng.uniform(0.5, 2.0)
                new_answer = str(round(float(answer) * scale, 2))
                new_prompt = prompt
                for num in numbers:
                    new_prompt = new_prompt.replace(
                        num, str(round(float(num) * scale, 2)), 1
                    )
            elif i < 4:
                # Time rescaling: d scales as t^2
                t_scale = rng.uniform(0.7, 1.5)
                new_prompt = prompt
                new_answer = str(round(float(answer) * t_scale * t_scale, 2))
                for j, num in enumerate(numbers):
                    if j % 2 == 0:  # Even indices = time values (heuristic)
                        new_prompt = new_prompt.replace(
                            num, str(round(float(num) * t_scale, 2)), 1
                        )
                    else:  # Odd indices = distance values
                        new_prompt = new_prompt.replace(
                            num, str(round(float(num) * t_scale * t_scale, 2)), 1
                        )
            else:
                # Unit variation: multiply all distances by 100 (m -> cm)
                scale = 100.0
                new_prompt = prompt.replace("meters", "centimeters").replace(
                    "m/s", "cm/s"
                )
                new_answer = str(round(float(answer) * scale, 2))
                for num in numbers:
                    new_prompt = new_prompt.replace(
                        num, str(round(float(num) * scale, 2)), 1
                    )

            augmented.append((new_prompt, new_answer))
        except (ValueError, ZeroDivisionError):
            continue

    return augmented


# ── Unit conversion augmentation (R+ scaling) ────────────────

def augment_unit_task(prompt, answer, n_augments=5):
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    numbers = re.findall(r'(\d+\.?\d*)', prompt)
    if len(numbers) < 2:
        return []

    for i in range(n_augments):
        try:
            if i < 3:
                # Scale conversion factor
                scale = rng.uniform(0.3, 3.0)
                new_answer = str(round(float(answer) * scale, 2))
                new_prompt = prompt
                for num in numbers:
                    new_prompt = new_prompt.replace(
                        num, str(round(float(num) * scale, 2)), 1
                    )
            else:
                # Rescale inputs only (keep conversion factor, change numbers)
                offset = rng.uniform(-10, 10)
                new_prompt = prompt
                new_answer = str(round(float(answer) + offset, 2))
                for num in numbers:
                    new_prompt = new_prompt.replace(
                        num, str(round(float(num) + offset, 2)), 1
                    )

            augmented.append((new_prompt, new_answer))
        except (ValueError, ZeroDivisionError):
            continue

    return augmented


# ── Numeral augmentation (nearby numbers + reordering) ───────

def augment_numeral_task(prompt, answer, n_augments=3):
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    # Try to find the target number
    target_num = None
    try:
        target_num = roman_to_int(answer)
    except Exception:
        try:
            target_num = int(answer)
        except ValueError:
            pass

    if target_num and target_num > 0:
        # Generate nearby number variants
        for _ in range(n_augments):
            delta = rng.choice([-5, -3, -1, 1, 3, 5, 10, -10])
            new_num = max(1, target_num + delta)

            # If answer was Roman, generate Roman for new number
            if all(c in "IVXLCDM" for c in answer.upper()):
                new_answer = int_to_roman(new_num)
                new_prompt = prompt.replace(str(target_num), str(new_num))
            else:
                new_answer = str(new_num)
                old_roman = int_to_roman(target_num)
                new_roman = int_to_roman(new_num)
                new_prompt = prompt.replace(old_roman, new_roman)

            if new_prompt != prompt:
                augmented.append((new_prompt, new_answer))
    else:
        # Fallback: example reordering
        lines = prompt.split("\n")
        example_lines = [l for l in lines if "->" in l or "=" in l]
        other_lines = [l for l in lines if l not in example_lines]
        if len(example_lines) >= 2:
            for _ in range(n_augments):
                rng.shuffle(example_lines)
                new_prompt = "\n".join(other_lines[:1] + example_lines + other_lines[1:])
                augmented.append((new_prompt, answer))

    return augmented


# ── Symbol augmentation (S_n permutation) ────────────────────

def augment_symbol_task(prompt, answer, n_augments=4):
    symbols = sorted(set(
        c for c in prompt + answer
        if c not in " =\n\t" and not c.isalnum()
    ))
    if len(symbols) < 3:
        return []

    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    for _ in range(n_augments):
        perm = list(symbols)
        rng.shuffle(perm)
        mapping = dict(zip(symbols, perm))

        def remap(text, m=mapping):
            result = list(text)
            temp = {}
            for i, c in enumerate(result):
                if c in m:
                    temp[i] = m[c]
            for i, c in temp.items():
                result[i] = c
            return "".join(result)

        augmented.append((remap(prompt), remap(answer)))

    return augmented


# ── Apply augmentation ───────────────────────────────────────

AUGMENT_FNS = {
    "bit": (augment_bit_task, 8),
    "encrypt": (augment_encrypt_task, 5),
    "physics": (augment_physics_task, 5),
    "unit": (augment_unit_task, 5),
    "numeral": (augment_numeral_task, 3),
    "symbol": (augment_symbol_task, 4),
}

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

print(f"Original: {len(tasks)}, After augmentation: {len(augmented_tasks)} "
      f"({len(augmented_tasks)/len(tasks):.1f}x)")
for t in sorted(aug_counts.keys()):
    print(f"  {t}: +{aug_counts[t]} augmented")

random.shuffle(augmented_tasks)


# ═══════════════════════════════════════════════════════════════
# STEP 3: Format training data with rich chain-of-thought
# ═══════════════════════════════════════════════════════════════

COT_TEMPLATES = {
    "bit": (
        "Let me analyze the bit transformation by examining each position.\n\n"
        "For each bit position i (0-7), I'll trace the input bit to the output bit "
        "across all examples to find the permutation rule.\n\n"
        "Position 0: input bits map to output position ?\n"
        "Position 1: input bits map to output position ?\n"
        "... checking each position against all examples ...\n\n"
        "The transformation is a permutation of bit positions.\n"
        "If there's also a complement (bit flip), I check whether "
        "output = NOT(permuted input).\n\n"
        "Applying the discovered rule to the query:\n\n"
    ),
    "encrypt": (
        "Let me decode the substitution cipher by building a character mapping.\n\n"
        "From example 1, I match each cipher character to its plaintext:\n"
        "  cipher -> plain: building the substitution table...\n\n"
        "From example 2, I confirm existing mappings and discover new ones:\n"
        "  Checking consistency with example 1...\n\n"
        "Full substitution table so far:\n"
        "  a->?, b->?, c->?, ... (filling in from all examples)\n\n"
        "Applying the substitution to the query text character by character:\n\n"
    ),
    "physics": (
        "I need to find the gravitational constant g from the experimental data.\n\n"
        "Using the free-fall equation: d = 0.5 * g * t^2\n"
        "Rearranging: g = 2 * d / t^2\n\n"
        "From example 1: g = 2 * d1 / t1^2 = ?\n"
        "From example 2: g = 2 * d2 / t2^2 = ?\n"
        "Checking consistency: all examples should give the same g.\n\n"
        "Average g = (sum of computed g values) / (number of examples)\n\n"
        "Now computing the answer for the query:\n"
        "  d = 0.5 * g * t_query^2 = ?\n\n"
    ),
    "unit": (
        "I need to find the conversion factor from the given examples.\n\n"
        "For each input-output pair:\n"
        "  factor = output_value / input_value\n\n"
        "Example 1: factor = out1 / in1 = ?\n"
        "Example 2: factor = out2 / in2 = ?\n"
        "Checking: all factors should be consistent.\n\n"
        "Conversion factor = average of computed factors\n\n"
        "Applying to query: answer = query_value * factor = ?\n\n"
    ),
    "numeral": (
        "I need to identify the numeral system from the examples.\n\n"
        "Looking at the input-output patterns:\n"
        "  Example 1: input -> output (what mapping does this suggest?)\n"
        "  Example 2: input -> output (consistent with the hypothesis?)\n\n"
        "This appears to be a conversion to/from a specific numeral system.\n"
        "Let me verify the rule against all examples.\n\n"
        "Applying the identified rule to the query:\n\n"
    ),
    "symbol": (
        "I need to find the symbol transformation rules.\n\n"
        "Building a mapping from input symbols to output symbols:\n"
        "  From example 1: symbol_A -> symbol_B, symbol_C -> symbol_D, ...\n"
        "  From example 2: confirming symbol_A -> symbol_B, new: symbol_E -> symbol_F\n\n"
        "Checking consistency across all examples.\n"
        "Complete mapping: {each input symbol -> its output symbol}\n\n"
        "Applying the mapping to the query, symbol by symbol:\n\n"
    ),
}


def format_training(task):
    cot = COT_TEMPLATES.get(
        task["type"],
        "Let me think through this step by step.\n\n"
    )
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


# ═══════════════════════════════════════════════════════════════
# STEP 4: Load model — EXACTLY following the official demo
# ═══════════════════════════════════════════════════════════════

import kagglehub
import mamba_ssm  # REQUIRED before model load
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

OUTPUT_DIR = "/kaggle/working"

MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)

print(f"Loading model from {MODEL_PATH}...")
t0 = time.time()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Model loaded in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════
# STEP 5: Hyperparameter proxy sweep (3 configs on 20% data)
# ═══════════════════════════════════════════════════════════════

from trl import SFTTrainer, SFTConfig

CONFIGS = [
    {"lr": 1e-4, "name": "low_lr"},
    {"lr": 1.5e-4, "name": "mid_lr"},
    {"lr": 2e-4, "name": "high_lr"},
]

LORA_RANK = 32
proxy_size = len(train_texts) // 5
proxy_train = train_texts[:proxy_size]
proxy_val = val_texts[:max(len(val_texts) // 3, 50)]

print(f"\n=== HP Proxy Sweep ({len(CONFIGS)} configs on {proxy_size} samples) ===")
best_loss = float("inf")
best_lr = 1.5e-4

for cfg in CONFIGS:
    print(f"\n--- Config: {cfg['name']} (LR={cfg['lr']}) ---")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=16,
        target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)

    proxy_train_ds = Dataset.from_dict({"text": proxy_train})
    proxy_val_ds = Dataset.from_dict({"text": proxy_val})

    proxy_args = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/proxy_{cfg['name']}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=cfg["lr"],
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=proxy_args,
        train_dataset=proxy_train_ds,
        eval_dataset=proxy_val_ds,
        tokenizer=tokenizer,
    )

    result = trainer.train()
    eval_result = trainer.evaluate()
    eval_loss = eval_result.get("eval_loss", float("inf"))
    print(f"  Train loss: {result.training_loss:.4f}, Eval loss: {eval_loss:.4f}")

    if eval_loss < best_loss:
        best_loss = eval_loss
        best_lr = cfg["lr"]

    # Remove adapter for next config
    model = base_model

print(f"\n=== Best config: LR={best_lr} (eval_loss={best_loss:.4f}) ===")


# ═══════════════════════════════════════════════════════════════
# STEP 6: Full training with best hyperparameters
# ═══════════════════════════════════════════════════════════════

print(f"\nFull training with LR={best_lr}...")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

n_train = len(train_texts)
n_epochs = 2
grad_accum = 16
steps_per_epoch = n_train // grad_accum
total_steps = n_epochs * steps_per_epoch
eval_steps = max(total_steps // 8, 50)

print(f"  Dataset: {n_train} examples (augmented)")
print(f"  Epochs: {n_epochs}, LR: {best_lr}")
print(f"  Effective batch: {grad_accum}, Total steps: ~{total_steps}")

training_args = SFTConfig(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=n_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
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
    bf16=True,
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
# STEP 7: Validation with self-consistency voting + error analysis
# ═══════════════════════════════════════════════════════════════

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
    """Classify the type of error for analysis."""
    if not predicted:
        return "empty_output"
    if check_match(predicted, actual):
        return "correct"
    if "boxed" in predicted or "\\" in predicted:
        return "format_error"
    try:
        p, a = float(predicted), float(actual)
        if abs(p - a) <= 1:
            return "off_by_one"
        if abs(p - a) / max(abs(a), 1e-10) < 0.05:
            return "rounding_error"
    except ValueError:
        pass
    if task_type in ("encrypt", "symbol"):
        if len(actual) > 0 and len(predicted) > 0:
            common = sum(1 for a, b in zip(actual, predicted) if a == b)
            if common / max(len(actual), len(predicted)) > 0.8:
                return "partial_match"
    if task_type == "bit" and len(predicted) == 8 and len(actual) == 8:
        diff = sum(1 for a, b in zip(actual, predicted) if a != b)
        if diff <= 2:
            return f"bit_error_{diff}"
    return "wrong_answer"


def generate_with_voting(model, tokenizer, prompt_text, n_samples=5, temp=0.3):
    """Generate N samples and return majority vote + agreement ratio."""
    answers = []
    for i in range(n_samples):
        inputs = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=2048
        ).to("cuda")
        with torch.no_grad():
            if i == 0:
                # First sample: greedy
                outputs = model.generate(
                    **inputs, max_new_tokens=512,
                    temperature=1.0, do_sample=False
                )
            else:
                outputs = model.generate(
                    **inputs, max_new_tokens=512,
                    temperature=temp, do_sample=True, top_p=0.9
                )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        answers.append(extract_boxed(response))

    # Majority vote
    counts = Counter(a for a in answers if a)
    if not counts:
        return answers[0] if answers else "", 0.0
    majority = counts.most_common(1)[0]
    agreement = majority[1] / len(answers)
    return majority[0], agreement


print("\n=== Validation with Self-Consistency Voting ===")
model.eval()

# Per-type tracking
type_correct = {}
type_total = {}
type_errors = {}
greedy_correct = 0
voting_correct = 0
total = 0

val_originals = [
    t for t in val_tasks
    if "_aug" not in str(t.get("id", ""))
][:50]

for task in val_originals:
    prompt_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{task['prompt']}\n\n"
        f"Think step by step. Put your final answer in \\boxed{{}}.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # Greedy (single pass)
    inputs = tokenizer(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=2048
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512,
            temperature=1.0, do_sample=False
        )
    greedy_response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    greedy_pred = extract_boxed(greedy_response)

    # Voting (5 samples)
    voted_pred, agreement = generate_with_voting(
        model, tokenizer, prompt_text, n_samples=5
    )

    actual = task["answer"]
    ttype = task.get("type", "unknown")

    greedy_match = check_match(greedy_pred, actual)
    voting_match = check_match(voted_pred, actual)

    if greedy_match:
        greedy_correct += 1
    if voting_match:
        voting_correct += 1

    # Per-type tracking (using voted result)
    type_total[ttype] = type_total.get(ttype, 0) + 1
    if voting_match:
        type_correct[ttype] = type_correct.get(ttype, 0) + 1
    else:
        if ttype not in type_errors:
            type_errors[ttype] = []
        type_errors[ttype].append((voted_pred, actual, task["prompt"][:60]))

    total += 1
    if total % 10 == 0:
        print(f"  [{total}/50] greedy={greedy_correct/total:.0%} "
              f"voting={voting_correct/total:.0%}")

print(f"\n=== RESULTS ===")
print(f"Greedy accuracy:  {greedy_correct}/{total} ({greedy_correct/total:.1%})")
print(f"Voting accuracy:  {voting_correct}/{total} ({voting_correct/total:.1%})")

print(f"\n=== PER-TYPE ACCURACY ===")
for ttype in sorted(type_total.keys()):
    tc = type_correct.get(ttype, 0)
    tt = type_total[ttype]
    print(f"  {ttype:12s}: {tc}/{tt} ({tc/max(tt,1):.0%})")

print(f"\n=== ERROR ANALYSIS ===")
for ttype in sorted(type_errors.keys()):
    errors = type_errors[ttype]
    error_modes = Counter()
    for pred, actual, _ in errors:
        error_modes[classify_error(pred, actual, ttype)] += 1
    print(f"\n  {ttype} ({len(errors)} errors):")
    for mode, count in error_modes.most_common():
        print(f"    {mode}: {count}")


# ═══════════════════════════════════════════════════════════════
# STEP 8: Save adapter and package submission
# ═══════════════════════════════════════════════════════════════

print("\nSaving adapter...")
model.save_pretrained(OUTPUT_DIR)

import subprocess
subprocess.run("zip -m submission.zip *", shell=True, check=True, cwd=OUTPUT_DIR)

print("\n" + "=" * 70)
print("SUBMISSION READY: /kaggle/working/submission.zip")
print(f"Augmentation: {len(tasks)} -> {len(augmented_tasks)} ({len(augmented_tasks)/len(tasks):.1f}x)")
print(f"Best LR: {best_lr}")
print(f"Greedy accuracy: {greedy_correct}/{total} ({greedy_correct/total:.1%})")
print(f"Voting accuracy: {voting_correct}/{total} ({voting_correct/total:.1%})")
print("=" * 70)
print("Done.")

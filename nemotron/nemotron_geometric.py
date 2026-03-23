"""Nemotron 3 Reasoning Challenge — Geometric Training Pipeline
NVIDIA Competition | $106K + DGX Sparks

Layer Cake Architecture:
  Layer 1: Vanilla baseline (official demo setup)
  Layer 2: Group-theoretic data augmentation (Bond, 2026a, Ch. 13)
  Layer 3: Symmetrized chain-of-thought prompting

Symmetry groups by task type:
  - Bit manipulation: Bit permutation S_8, complement Z_2, circular shift Z_8
  - Encryption: Alphabet permutation (consistent relabeling of cipher)
  - Physics: Scale invariance (rescale g, recompute all examples)
  - Unit conversion: Affine group (rescale conversion factor)
  - Numeral system: Identity (well-defined mapping, augment with nearby numbers)
  - Symbol transform: Symbol permutation group

Reference: "Geometric Methods in Computational Modeling" (Bond, 2026a)
  Ch. 13: Group-Theoretic Data Augmentation
  Ch. 13.3: Consistent Augmentation of Input-Output Pairs
  Ch. 13.10: When to Augment / How Many Augmentations

Paste this ENTIRE file into ONE cell in a Kaggle notebook.
Enable GPU (T4 or P100). Add competition data as input.
Expected runtime: ~2-3 hours with augmentation.
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Setup
# ═══════════════════════════════════════════════════════════════

# (pip installs and unsloth import moved to top of script)

import os
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
os.environ["UNSLOTH_COMPILE_CACHE"] = "/tmp/unsloth_cache"
os.makedirs("/tmp/unsloth_cache", exist_ok=True)

import sys
!pip install -q --target=/tmp/pylibs --no-deps trl unsloth unsloth_zoo bitsandbytes
sys.path.insert(0, "/tmp/pylibs")

# Import unsloth FIRST (before transformers/peft)
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("Unsloth loaded successfully")
except Exception as e:
    USE_UNSLOTH = False
    print(f"Unsloth not available: {e}")

import json, time, random, re, math
import polars as pl
import torch
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    print("WARNING: No GPU. Enable GPU in notebook settings.")

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# CELL 2: Load Data
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/nvidia-nemotron-3-reasoning-challenge"
if not os.path.exists(DATA_DIR):
    # Fallback: search for train.csv
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            DATA_DIR = root
            break

train_df = pl.read_csv(f"{DATA_DIR}/train.csv")
test_df = pl.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")


# ═══════════════════════════════════════════════════════════════
# LAYER 1: Task Classification
# ═══════════════════════════════════════════════════════════════

def classify_task(prompt):
    """Classify prompt into one of 6 task types."""
    if "bit manipulation" in prompt:
        return "bit"
    elif "encryption" in prompt or "decrypt" in prompt:
        return "encrypt"
    elif "gravitational" in prompt:
        return "physics"
    elif "unit conversion" in prompt:
        return "unit"
    elif "numeral system" in prompt:
        return "numeral"
    elif "transformation rules" in prompt:
        return "symbol"
    return "unknown"


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


# ═══════════════════════════════════════════════════════════════
# LAYER 2: Group-Theoretic Data Augmentation
#
# "Symmetry is not decoration — it is a computational resource."
#   — Bond (2026a), Ch. 13.11
#
# For each task type, we identify the symmetry group that acts
# on the input-output pairs, and augment by applying group
# elements CONSISTENTLY to both inputs and outputs within each
# prompt. This is the key insight from Ch. 13.3.3.
# ═══════════════════════════════════════════════════════════════

# --- Bit Manipulation Augmentation ---
# Symmetry group: bit permutation (S_8) × complement (Z_2)
# Action: permute bit positions consistently across ALL examples in prompt

def parse_bit_examples(prompt):
    """Extract input->output pairs and the query from a bit manipulation prompt."""
    lines = prompt.split("\n")
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if "->" in line and len(line) < 30:
            parts = line.split("->")
            if len(parts) == 2:
                inp = parts[0].strip()
                out = parts[1].strip()
                if len(inp) == 8 and len(out) == 8 and all(c in "01" for c in inp + out):
                    examples.append((inp, out))
        elif "output for:" in line.lower() or "determine the output" in line.lower():
            # Extract the query bits
            match = re.search(r'([01]{8})', line)
            if match:
                query = match.group(1)
    # Also check last line for query
    if query is None:
        for line in reversed(lines):
            match = re.search(r'([01]{8})\s*$', line.strip())
            if match:
                query = match.group(1)
                break
    return examples, query


def apply_bit_permutation(bitstring, perm):
    """Apply a bit position permutation to an 8-bit string."""
    return "".join(bitstring[p] for p in perm)


def apply_bit_complement(bitstring):
    """Flip all bits (Z_2 complement action)."""
    return "".join("1" if b == "0" else "0" for b in bitstring)


def augment_bit_task(prompt, answer, n_augments=3):
    """Generate n augmented versions of a bit manipulation task.

    Group: S_8 (bit permutation) × Z_2 (complement)
    Following Bond (2026a) Ch. 13.3.3: same transform applied to
    ALL input-output pairs AND the query, consistently.
    """
    examples, query = parse_bit_examples(prompt)
    if not examples or not query:
        return []

    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    for _ in range(n_augments):
        # Sample random permutation from S_8
        perm = list(range(8))
        rng.shuffle(perm)

        # Coin flip for complement (Z_2)
        do_complement = rng.random() > 0.5

        # Apply consistently to ALL examples + query + answer
        new_examples = []
        for inp, out in examples:
            new_inp = apply_bit_permutation(inp, perm)
            new_out = apply_bit_permutation(out, perm)
            if do_complement:
                new_inp = apply_bit_complement(new_inp)
                new_out = apply_bit_complement(new_out)
            new_examples.append((new_inp, new_out))

        new_query = apply_bit_permutation(query, perm)
        new_answer = apply_bit_permutation(answer, perm)
        if do_complement:
            new_query = apply_bit_complement(new_query)
            new_answer = apply_bit_complement(new_answer)

        # Reconstruct prompt
        ex_lines = "\n".join(f"{inp} -> {out}" for inp, out in new_examples)
        new_prompt = (
            "In Alice's Wonderland, a secret bit manipulation rule transforms "
            "8-bit binary numbers. The transformation involves operations like "
            "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority "
            "or choice functions.\n\n"
            "Here are some examples of input -> output:\n"
            f"{ex_lines}\n\n"
            f"Now, determine the output for: {new_query}"
        )
        augmented.append((new_prompt, new_answer))

    return augmented


# --- Encryption Augmentation ---
# Symmetry group: consistent relabeling of cipher alphabet
# If the cipher is a substitution, relabeling both cipher and plain
# sides consistently preserves the mapping structure.

def parse_encrypt_examples(prompt):
    """Extract cipher->plain pairs and query from encryption prompt."""
    lines = prompt.split("\n")
    examples = []
    query_text = None
    for line in lines:
        line = line.strip()
        if " -> " in line and len(line) > 10:
            parts = line.split(" -> ", 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        elif "decrypt the following" in line.lower():
            # Query is after the colon
            match = re.search(r':\s*(.+)$', line)
            if match:
                query_text = match.group(1).strip()
    # Check next line after "decrypt" if not found
    if query_text is None:
        for i, line in enumerate(lines):
            if "decrypt" in line.lower() and i + 1 < len(lines):
                query_text = lines[i + 1].strip()
                break
    return examples, query_text


def build_char_map(examples):
    """Build cipher->plain character mapping from examples."""
    cipher_to_plain = {}
    for cipher, plain in examples:
        c_words = cipher.split()
        p_words = plain.split()
        if len(c_words) == len(p_words):
            for cw, pw in zip(c_words, p_words):
                if len(cw) == len(pw):
                    for cc, pc in zip(cw, pw):
                        if cc in cipher_to_plain and cipher_to_plain[cc] != pc:
                            pass  # inconsistency, skip
                        cipher_to_plain[cc] = pc
    return cipher_to_plain


def augment_encrypt_task(prompt, answer, n_augments=2):
    """Augment by consistently relabeling the plaintext alphabet.

    Group: S_26 acting on plaintext alphabet
    Apply same permutation to ALL plaintext sides of examples AND the answer.
    The cipher side stays the same (the model learns cipher->new_plain).
    """
    examples, query = parse_encrypt_examples(prompt)
    if not examples or not query or len(examples) < 3:
        return []

    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    for _ in range(n_augments):
        # Create a random permutation of the alphabet
        perm_alpha = list(alphabet)
        rng.shuffle(perm_alpha)
        char_map = {a: b for a, b in zip(alphabet, perm_alpha)}

        def remap_plain(text):
            return "".join(char_map.get(c, c) for c in text)

        # Apply to plaintext side of all examples
        new_examples = []
        for cipher, plain in examples:
            new_examples.append((cipher, remap_plain(plain)))

        new_answer = remap_plain(answer)

        # Reconstruct
        ex_lines = "\n".join(f"{c} -> {p}" for c, p in new_examples)
        new_prompt = (
            "In Alice's Wonderland, secret encryption rules are used on text. "
            "Here are some examples:\n"
            f"{ex_lines}\n"
            f"Now, decrypt the following text: {query}"
        )
        augmented.append((new_prompt, new_answer))

    return augmented


# --- Physics Augmentation ---
# Symmetry group: scale invariance (rescale g)
# d = 0.5 * g * t^2, so scaling g by factor k scales all distances by k.

def parse_physics_examples(prompt):
    """Extract (t, d) pairs and query t from physics prompt."""
    examples = []
    query_t = None
    for match in re.finditer(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)', prompt):
        examples.append((float(match.group(1)), float(match.group(2))))
    match = re.search(r'for\s+t\s*=\s*([\d.]+)\s*s\s+given', prompt)
    if match:
        query_t = float(match.group(1))
    return examples, query_t


def augment_physics_task(prompt, answer, n_augments=2):
    """Augment by rescaling g (and thus all distances).

    Group: R+ (positive reals under multiplication) acting on g.
    d = 0.5 * g * t^2, so scaling g by k scales all d by k.
    Times stay the same. This teaches the model the algebraic
    structure rather than memorizing specific g values.
    """
    examples, query_t = parse_physics_examples(prompt)
    if not examples or query_t is None:
        return []

    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    for _ in range(n_augments):
        scale = rng.uniform(0.5, 2.0)  # Scale g by random factor

        new_examples = [(t, round(d * scale, 2)) for t, d in examples]
        new_answer = round(float(answer) * scale, 2)

        ex_lines = "\n".join(f"For t = {t:.2f}s, distance = {d:.2f} m"
                             for t, d in new_examples)
        new_prompt = (
            "In Alice's Wonderland, the gravitational constant has been "
            "secretly changed. Here are some example observations:\n"
            f"{ex_lines}\n"
            f"Now, determine the falling distance for t = {query_t:.2f}s "
            f"given d = 0.5*g*t^2."
        )
        augmented.append((new_prompt, str(new_answer)))

    return augmented


# --- Unit Conversion Augmentation ---
# Symmetry group: affine transformations on the conversion factor
# If original conversion is y = a*x + b, rescaling gives y' = k*a*x + k*b

def parse_unit_examples(prompt):
    """Extract (input, output) measurement pairs."""
    examples = []
    query_val = None
    for match in re.finditer(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt):
        examples.append((float(match.group(1)), float(match.group(2))))
    match = re.search(r'convert.*?:\s*([\d.]+)\s*m', prompt)
    if match:
        query_val = float(match.group(1))
    return examples, query_val


def augment_unit_task(prompt, answer, n_augments=2):
    """Augment by rescaling the conversion factor.

    Group: R+ acting on the output scale.
    If y = f(x), then k*y = k*f(x) is a valid augmented conversion.
    """
    examples, query_val = parse_unit_examples(prompt)
    if not examples or query_val is None:
        return []

    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    for _ in range(n_augments):
        scale = rng.uniform(0.7, 1.5)

        new_examples = [(x, round(y * scale, 2)) for x, y in examples]
        new_answer = round(float(answer) * scale, 2)

        ex_lines = "\n".join(f"{x:.2f} m becomes {y:.2f}"
                             for x, y in new_examples)
        new_prompt = (
            "In Alice's Wonderland, a secret unit conversion is applied to "
            "measurements. For example:\n"
            f"{ex_lines}\n"
            f"Now, convert the following measurement: {query_val:.2f} m"
        )
        augmented.append((new_prompt, str(new_answer)))

    return augmented


# --- Numeral System Augmentation ---
# No natural group action that preserves the mapping.
# Augment by generating additional examples from the same system.

def augment_numeral_task(prompt, answer, n_augments=1):
    """Augment with additional number-numeral pairs from the same system.

    For Roman numerals: generate nearby numbers to reinforce the pattern.
    This is 'soft' augmentation — not group-theoretic, but still useful.
    """
    # Roman numeral tasks are well-defined; the model mostly needs to
    # see enough examples. We add paraphrased versions.
    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    # Slight reordering of examples
    lines = prompt.split("\n")
    example_lines = [l for l in lines if " -> " in l and len(l.strip()) < 30]
    if len(example_lines) < 2:
        return []

    for _ in range(n_augments):
        shuffled = list(example_lines)
        rng.shuffle(shuffled)
        # Rebuild with shuffled examples
        new_prompt = prompt
        for old, new in zip(example_lines, shuffled):
            new_prompt = new_prompt.replace(old, "PLACEHOLDER_" + old, 1)
        for i, old in enumerate(example_lines):
            new_prompt = new_prompt.replace("PLACEHOLDER_" + old, shuffled[i], 1)
        augmented.append((new_prompt, answer))

    return augmented


# --- Symbol Transform Augmentation ---
# Symmetry group: permutation of the symbol set
# Relabel ALL symbols consistently across examples + query + answer.

def get_symbol_set(text):
    """Extract the set of non-alphanumeric, non-space, non-equals symbols."""
    symbols = set()
    for c in text:
        if c not in " =\n\t" and not c.isalnum():
            symbols.add(c)
    return sorted(symbols)


def augment_symbol_task(prompt, answer, n_augments=2):
    """Augment by consistently permuting the symbol alphabet.

    Group: S_n acting on the symbol set.
    Apply same permutation to ALL examples, query, AND answer.
    This is directly analogous to color permutation in ARC-AGI
    (Bond 2026a, Ch. 13.7.1).
    """
    symbols = get_symbol_set(prompt + answer)
    if len(symbols) < 3:
        return []

    augmented = []
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)

    for _ in range(n_augments):
        perm = list(symbols)
        rng.shuffle(perm)
        sym_map = {s: p for s, p in zip(symbols, perm)}

        def remap(text):
            # Two-pass to avoid collisions: map to temp first
            result = list(text)
            for i, c in enumerate(result):
                if c in sym_map:
                    result[i] = f"\x00{symbols.index(c):02d}"
            result = "".join(result)
            for i, s in enumerate(symbols):
                result = result.replace(f"\x00{i:02d}", perm[i])
            return result

        new_prompt = remap(prompt)
        new_answer = remap(answer)
        augmented.append((new_prompt, new_answer))

    return augmented


# ═══════════════════════════════════════════════════════════════
# LAYER 2: Apply Augmentation Pipeline
# ═══════════════════════════════════════════════════════════════

AUGMENT_CONFIG = {
    "bit": {"fn": augment_bit_task, "n": 3},       # S_8 × Z_2 = huge group, 3 samples
    "encrypt": {"fn": augment_encrypt_task, "n": 2}, # S_26, 2 samples
    "physics": {"fn": augment_physics_task, "n": 2}, # R+, 2 samples
    "unit": {"fn": augment_unit_task, "n": 2},       # R+, 2 samples
    "numeral": {"fn": augment_numeral_task, "n": 1}, # example reordering only
    "symbol": {"fn": augment_symbol_task, "n": 2},   # S_n, 2 samples
}

print("\n" + "=" * 70)
print("LAYER 2: Group-Theoretic Data Augmentation")
print("=" * 70)

augmented_tasks = []
aug_counts = {t: 0 for t in AUGMENT_CONFIG}
aug_failures = {t: 0 for t in AUGMENT_CONFIG}

for task in tasks:
    # Always include original
    augmented_tasks.append(task)

    # Generate augmented versions
    ttype = task["type"]
    if ttype in AUGMENT_CONFIG:
        cfg = AUGMENT_CONFIG[ttype]
        try:
            augs = cfg["fn"](task["prompt"], task["answer"], cfg["n"])
            for new_prompt, new_answer in augs:
                augmented_tasks.append({
                    "id": task["id"] + "_aug",
                    "prompt": new_prompt,
                    "answer": new_answer,
                    "type": ttype,
                })
                aug_counts[ttype] += 1
        except Exception as e:
            aug_failures[ttype] += 1

total_orig = len(tasks)
total_aug = len(augmented_tasks)
print(f"Original: {total_orig}")
print(f"After augmentation: {total_aug} ({total_aug/total_orig:.1f}x)")
print(f"Augmentation by type:")
for t in sorted(aug_counts.keys()):
    tc = type_counts.get(t, 0)
    ac = aug_counts[t]
    af = aug_failures[t]
    print(f"  {t:12s}: {tc:5d} orig + {ac:5d} aug = {tc+ac:5d} total"
          f"  ({ac/max(tc,1):.1f}x lift)"
          f"{'  [' + str(af) + ' failures]' if af else ''}")

random.shuffle(augmented_tasks)


# ═══════════════════════════════════════════════════════════════
# LAYER 3: Format Training Data with Chain-of-Thought
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("LAYER 3: Chain-of-Thought Prompt Formatting")
print("=" * 70)

# Task-specific CoT templates that guide structured reasoning
COT_TEMPLATES = {
    "bit": (
        "Let me analyze the bit transformation pattern.\n"
        "I'll examine each bit position to find the rule.\n\n"
    ),
    "encrypt": (
        "Let me decode the substitution cipher.\n"
        "I'll map each cipher character to its plaintext equivalent.\n\n"
    ),
    "physics": (
        "I need to find g from the data using d = 0.5*g*t^2.\n"
        "From the examples: g = 2*d/t^2.\n\n"
    ),
    "unit": (
        "I need to find the conversion factor from the examples.\n"
        "Let me compute output/input for each pair.\n\n"
    ),
    "numeral": (
        "I need to identify the numeral system and apply it.\n"
        "Let me analyze the mapping pattern.\n\n"
    ),
    "symbol": (
        "I need to find the symbol transformation rules.\n"
        "Let me map each input symbol to its output.\n\n"
    ),
}

def format_training_example(task):
    """Format a task as a training example with CoT."""
    ttype = task["type"]
    cot = COT_TEMPLATES.get(ttype, "Let me think through this step by step.\n\n")

    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{task['prompt']}\n\n"
        f"Think step by step. Put your final answer in \\boxed{{}}.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{cot}"
        f"\\boxed{{{task['answer']}}}\n"
        f"<|eot_id|>"
    )
    return text


formatted = [format_training_example(t) for t in augmented_tasks]

# Split 90/10
split_idx = int(len(formatted) * 0.9)
train_texts = formatted[:split_idx]
val_texts = formatted[split_idx:]
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
print(f"Avg prompt length: {sum(len(t) for t in formatted) / len(formatted):.0f} chars")


# ═══════════════════════════════════════════════════════════════
# CELL 3: Load Model (Official Demo Setup)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Loading Nemotron-3-Nano-30B")
print("=" * 70)

from datasets import Dataset

OUTPUT_DIR = "/kaggle/working"
LORA_RANK = 32

# --- Try Unsloth (fast 4-bit, Kaggle-optimized) ---
if USE_UNSLOTH:
  try:
    print("Using Unsloth (4-bit quantized, optimized)")
    t0 = time.time()
    os.makedirs("/tmp/offload", exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
        max_seq_length=2048,
        load_in_4bit=True,
        offload_folder="/tmp/offload",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "in_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    print(f"Model loaded via Unsloth in {time.time()-t0:.0f}s")

  except Exception as e:
    raise RuntimeError(
        f"Unsloth model load failed: {e}\n"
        f"This model requires 4-bit quantization via Unsloth to fit in GPU memory."
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.print_trainable_parameters()


# ═══════════════════════════════════════════════════════════════
# CELL 4: Configure LoRA (Rank 32, official target modules)
# ═══════════════════════════════════════════════════════════════

# LoRA already configured above (Unsloth or peft fallback)


# ═══════════════════════════════════════════════════════════════
# CELL 5: Train
# ═══════════════════════════════════════════════════════════════

from trl import SFTTrainer, SFTConfig

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

# Compute effective batch size
# With gradient_accumulation_steps=16 and batch_size=1:
# effective_batch = 16 — good for augmented dataset
n_train = len(train_texts)
n_epochs = 2  # fewer epochs since augmented data provides more diversity
steps_per_epoch = n_train // 16  # effective batch = 16
total_steps = n_epochs * steps_per_epoch
eval_steps = max(total_steps // 10, 50)

print(f"\nTraining config:")
print(f"  Dataset: {n_train} examples ({total_orig} original + augmented)")
print(f"  Epochs: {n_epochs}")
print(f"  Effective batch size: 16")
print(f"  Steps/epoch: {steps_per_epoch}")
print(f"  Total steps: ~{total_steps}")
print(f"  Eval every: {eval_steps} steps")

training_args = SFTConfig(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=n_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
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
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
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
# CELL 6: Quick Validation
# ═══════════════════════════════════════════════════════════════

def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def check_answer(predicted, actual):
    """Check if predicted matches actual (string or numeric tolerance)."""
    if predicted.strip() == actual.strip():
        return True
    try:
        p, a = float(predicted), float(actual)
        if abs(a) < 1e-10:
            return abs(p) < 1e-10
        return abs(p - a) / abs(a) < 0.01
    except (ValueError, ZeroDivisionError):
        return False


print("\nValidating on held-out set...")
model.eval()
correct = 0
total = 0
type_correct = {}
type_total = {}

# Use original (non-augmented) validation examples only
val_originals = [t for t in augmented_tasks[split_idx:] if "_aug" not in str(t["id"])]
val_sample = val_originals[:100]

for task in val_sample:
    prompt_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{task['prompt']}\n\n"
        f"Think step by step. Put your final answer in \\boxed{{}}.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512,
            temperature=0.0, top_p=1.0, do_sample=False,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    predicted = extract_boxed_answer(response)
    actual = task["answer"]
    is_correct = check_answer(predicted, actual)

    total += 1
    ttype = task["type"]
    type_total[ttype] = type_total.get(ttype, 0) + 1

    if is_correct:
        correct += 1
        type_correct[ttype] = type_correct.get(ttype, 0) + 1
    elif total <= 5:
        print(f"  MISS [{ttype}]: pred='{predicted[:40]}' actual='{actual[:40]}'")

    if total % 20 == 0:
        print(f"  [{total}/{len(val_sample)}] accuracy: {correct/total:.0%}")

print(f"\nOverall validation accuracy: {correct}/{total} ({correct/total:.1%})")
print(f"\nPer-type accuracy:")
for t in sorted(type_total.keys()):
    tc = type_correct.get(t, 0)
    tt = type_total[t]
    print(f"  {t:12s}: {tc}/{tt} ({tc/max(tt,1):.0%})")


# ═══════════════════════════════════════════════════════════════
# CELL 7: Save Adapter & Package Submission
# ═══════════════════════════════════════════════════════════════

print("\nSaving adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Verify required files
assert os.path.exists(f"{OUTPUT_DIR}/adapter_config.json"), "Missing adapter_config.json!"
assert os.path.exists(f"{OUTPUT_DIR}/adapter_model.safetensors"), "Missing adapter weights!"

import subprocess
subprocess.run(
    f"cd {OUTPUT_DIR} && zip -m /kaggle/working/submission.zip "
    f"adapter_config.json adapter_model.safetensors",
    shell=True, check=True,
)

print("\n" + "=" * 70)
print("SUBMISSION READY: /kaggle/working/submission.zip")
print("=" * 70)
print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")
print(f"Augmentation: {total_aug - total_orig} additional examples "
      f"({total_aug/total_orig:.1f}x)")
print(f"Validation accuracy: {correct}/{total} ({correct/total:.1%})")
print(f"\nGeometric Methods Applied:")
print(f"  Bit manipulation: S_8 × Z_2 (bit permutation + complement)")
print(f"  Encryption: S_26 (alphabet permutation)")
print(f"  Physics: R+ (gravitational scale invariance)")
print(f"  Unit conversion: R+ (conversion factor scaling)")
print(f"  Numeral system: example reordering")
print(f"  Symbol transform: S_n (symbol permutation)")
print(f"\nReference: Bond (2026a), 'Geometric Methods in Computational")
print(f"  Modeling', Ch. 13: Group-Theoretic Data Augmentation")

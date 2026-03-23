"""Nemotron 3 Reasoning Challenge — Atlas GV100 Training Script.

Optimized for 2x Quadro GV100 (32GB each, Volta, compute 7.0).
- Full LoRA rank 32 (competition maximum)
- FP16 mixed precision (Volta Tensor Cores)
- No quantization needed (64GB total GPU RAM)
- device_map="auto" across both GPUs
"""

import os, time, random, re, shutil
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

WORK_DIR = Path(os.environ.get("WORK_DIR", os.path.expanduser("~/nemotron")))
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({vram:.1f} GB)")
else:
    raise RuntimeError("No GPU detected")

# GV100 supports fp16 Tensor Cores but NOT bf16
USE_FP16 = True
COMPUTE_DTYPE = torch.float16
print(f"  Using fp16 (Volta Tensor Cores)")

# ═══════════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════════

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

# ═══════════════════════════════════════════════════════════════
# Format Training Data
# ═══════════════════════════════════════════════════════════════

def format_prompt(prompt, answer=None):
    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}\n\n"
        f"Think step by step. Put your final answer in \\boxed{{}}.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if answer is not None:
        text += f"After analyzing the pattern, the answer is:\n\n\\boxed{{{answer}}}\n<|eot_id|>"
    return text

formatted = []
for _, row in train_df.iterrows():
    formatted.append({
        "text": format_prompt(row["prompt"], row["answer"]),
        "prompt": row["prompt"],
        "answer": str(row["answer"]),
    })

random.seed(42)
random.shuffle(formatted)
split_idx = int(len(formatted) * 0.9)
train_dataset = Dataset.from_list(formatted[:split_idx])
val_dataset = Dataset.from_list(formatted[split_idx:])
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ═══════════════════════════════════════════════════════════════
# Load Model — NO quantization on GV100 (32GB per GPU = 64GB total)
# ═══════════════════════════════════════════════════════════════

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"

print(f"\nLoading {MODEL_NAME} in fp16 across 2x GV100...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=COMPUTE_DTYPE,
)
print(f"Model loaded in {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# Configure LoRA — Full rank 32 (competition maximum)
# ═══════════════════════════════════════════════════════════════

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# Train — Volta-optimized settings
# ═══════════════════════════════════════════════════════════════

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Can fit batch=2 with 32GB per GPU
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch = 2*2*4 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    fp16=True,      # Volta Tensor Cores
    bf16=False,     # Volta does NOT support bf16
    gradient_checkpointing=True,
    max_seq_length=2048,  # Full context — 32GB GPUs can handle it
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
print(f"Training complete in {elapsed:.1f} min, loss: {result.training_loss:.4f}")

ADAPTER_DIR = OUTPUT_DIR / "final_adapter"
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"Adapter saved to {ADAPTER_DIR}")

# ═══════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════

def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""

print("\nEvaluating on validation set...")
model.eval()
correct = 0
total_eval = 0

for item in formatted[split_idx:split_idx+50]:
    prompt_text = format_prompt(item["prompt"])
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, top_p=1.0, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    predicted = extract_boxed_answer(response)
    actual = item["answer"]
    total_eval += 1
    match = predicted.strip() == actual.strip()
    if not match:
        try:
            match = abs(float(predicted) - float(actual)) / max(abs(float(actual)), 1e-10) < 0.01
        except (ValueError, ZeroDivisionError):
            pass
    if match:
        correct += 1
    elif total_eval <= 10:
        print(f"  MISS: predicted='{predicted[:30]}' actual='{actual[:30]}'")
    if total_eval % 10 == 0:
        print(f"  [{total_eval}/50] accuracy: {correct/total_eval:.0%}")

print(f"\nValidation accuracy: {correct}/{total_eval} ({correct/total_eval:.1%})")

# ═══════════════════════════════════════════════════════════════
# Package Submission
# ═══════════════════════════════════════════════════════════════

SUBMISSION_DIR = OUTPUT_DIR / "submission_adapter"
SUBMISSION_DIR.mkdir(exist_ok=True)
for f in ADAPTER_DIR.iterdir():
    shutil.copy2(f, SUBMISSION_DIR / f.name)

submission_zip = OUTPUT_DIR / "submission"
shutil.make_archive(str(submission_zip), "zip", str(SUBMISSION_DIR))
print(f"\n{submission_zip}.zip created")
print(f"\n=== Done! ===")

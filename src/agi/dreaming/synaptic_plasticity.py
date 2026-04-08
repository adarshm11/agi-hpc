# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Synaptic Plasticity — LoRA fine-tuning from dreaming consolidation.

After the dreaming pipeline synthesizes wiki articles, this module
converts them into instruction-tuning pairs and runs a short LoRA
fine-tuning session on the Ego (Gemma 4 E4B). This is the difference
between "remembering" (wiki articles improve RAG) and "learning"
(model weights actually change).

Biological grounding:
    - Tononi & Cirelli (2006): Synaptic Homeostasis Hypothesis —
      sleep downscales synapses that were strengthened during wake,
      improving signal-to-noise ratio
    - McClelland et al. (1995): Complementary Learning Systems —
      new experiences (hippocampal/episodic) are slowly integrated
      into neocortical (parametric) knowledge during sleep

Pipeline position:
    Stage 1: Episodic Replay
    Stage 2: Topic Clustering
    Stage 3: Article Synthesis (certainty scoring)
    Stage 4: Creative Dreaming
    Stage 5: Housekeeping
    Stage 5b: Synaptic Plasticity (THIS MODULE)
        → Extract instruction/output pairs from wiki articles
        → Run short LoRA fine-tune on Ego (Gemma 4 E4B)
        → Export merged GGUF for Ollama/llama-server reload

Constraints:
    - Runs on GPU 1 (must stop Id inference first, or use CPU)
    - Short sessions only: 1 epoch, ~50-200 examples, ~10 minutes
    - Catastrophic forgetting mitigation: low learning rate, small rank
    - Only runs when curriculum score > 0.7 (don't reinforce bad habits)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlasticityConfig:
    """Configuration for the synaptic plasticity module.

    Attributes:
        wiki_dir: Path to wiki articles.
        output_dir: Path for LoRA adapter output.
        model_name: Base model for fine-tuning.
        gpu_index: GPU to use (1 = Id's GPU, freed during dreaming).
        max_examples: Maximum instruction pairs per session.
        learning_rate: Conservative LR to prevent catastrophic forgetting.
        lora_rank: Low rank for incremental updates.
        min_score_threshold: Minimum curriculum score to allow plasticity.
        max_seq_length: Maximum sequence length for training.
    """

    wiki_dir: str = "/home/claude/agi-hpc/wiki"
    output_dir: str = "/home/claude/models/gemma4-ego-plasticity"
    model_name: str = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    gpu_index: int = 1
    max_examples: int = 200
    learning_rate: float = 5e-5  # Conservative: 4x lower than initial training
    lora_rank: int = 8  # Lower rank = smaller updates = less forgetting
    lora_alpha: int = 16
    min_score_threshold: float = 0.7
    max_seq_length: int = 1024
    num_epochs: int = 1  # Single epoch — incremental, not overfit


@dataclass
class InstructionPair:
    """A single instruction-tuning example extracted from a wiki article."""

    instruction: str
    output: str
    source_article: str
    certainty_grade: str  # A, B, C, D


@dataclass
class PlasticityResult:
    """Result of a synaptic plasticity session."""

    examples_generated: int
    examples_trained: int
    training_loss: float
    duration_seconds: float
    adapter_path: str
    skipped_reason: Optional[str] = None


def extract_instruction_pairs(
    wiki_dir: str,
    max_examples: int = 200,
    min_grade: str = "B",
) -> List[InstructionPair]:
    """Extract instruction-tuning pairs from dream-consolidated wiki articles.

    Only extracts from articles with certainty grade >= min_grade to avoid
    training on uncertain knowledge (catastrophic forgetting mitigation).

    Args:
        wiki_dir: Path to wiki directory containing dream-*.md files.
        max_examples: Maximum number of pairs to extract.
        min_grade: Minimum article grade to include (A, B, C, D).

    Returns:
        List of InstructionPair objects.
    """
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    min_grade_val = grade_order.get(min_grade, 1)

    wiki_path = Path(wiki_dir)
    if not wiki_path.exists():
        logger.warning("[plasticity] Wiki dir not found: %s", wiki_dir)
        return []

    articles = sorted(wiki_path.glob("dream-*.md"))
    if not articles:
        logger.info("[plasticity] No dream articles found")
        return []

    pairs: List[InstructionPair] = []

    for article_path in articles:
        try:
            content = article_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # Extract grade from certainty report
        grade = _extract_grade(content)
        grade_val = grade_order.get(grade, 3)

        if grade_val > min_grade_val:
            continue  # Skip low-certainty articles

        # Extract structured knowledge as instruction pairs
        article_pairs = _article_to_pairs(content, article_path.stem, grade)
        pairs.extend(article_pairs)

        if len(pairs) >= max_examples:
            break

    pairs = pairs[:max_examples]
    logger.info(
        "[plasticity] Extracted %d instruction pairs from %d articles",
        len(pairs),
        len(articles),
    )
    return pairs


def _extract_grade(content: str) -> str:
    """Extract the certainty grade from a wiki article."""
    for line in content.split("\n"):
        lower = line.lower()
        if "grade" in lower and any(
            g in line for g in ["**A**", "**B**", "**C**", "**D**"]
        ):
            for grade in ["A", "B", "C", "D"]:
                if f"**{grade}**" in line:
                    return grade
        # Also check for "Article Grade: A" pattern
        if "article grade" in lower:
            for grade in ["A", "B", "C", "D"]:
                if grade in line.split(":")[-1]:
                    return grade
    return "C"  # Default to C if not found


def _article_to_pairs(
    content: str,
    slug: str,
    grade: str,
) -> List[InstructionPair]:
    """Convert a wiki article into instruction-tuning pairs.

    Generates multiple pair types:
    1. Summary Q&A: "What is X?" → article summary
    2. Key concept Q&A: "Explain [concept]" → concept detail
    3. Procedure Q&A: "How do you [procedure]?" → steps
    """
    pairs: List[InstructionPair] = []
    sections = _parse_sections(content)

    # Title-based summary pair
    title = sections.get("title", slug.replace("dream-", "").replace("-", " "))
    summary = sections.get("summary", "")
    if summary:
        pairs.append(
            InstructionPair(
                instruction=f"What do you know about {title}?",
                output=summary,
                source_article=slug,
                certainty_grade=grade,
            )
        )

    # Key concepts
    concepts = sections.get("key_concepts", "")
    if concepts:
        pairs.append(
            InstructionPair(
                instruction=(f"What are the key concepts related to {title}?"),
                output=concepts,
                source_article=slug,
                certainty_grade=grade,
            )
        )

    # Detailed explanation
    detail = sections.get("detail", "")
    if detail:
        pairs.append(
            InstructionPair(
                instruction=f"Explain {title} in detail.",
                output=detail[:800],  # Truncate for training
                source_article=slug,
                certainty_grade=grade,
            )
        )

    # Procedures
    procedures = sections.get("procedures", "")
    if procedures:
        pairs.append(
            InstructionPair(
                instruction=f"What is the procedure for {title}?",
                output=procedures,
                source_article=slug,
                certainty_grade=grade,
            )
        )

    return pairs


def _parse_sections(content: str) -> Dict[str, str]:
    """Parse a wiki article into named sections."""
    sections: Dict[str, str] = {}
    current_section = "preamble"
    current_lines: List[str] = []

    for line in content.split("\n"):
        if line.startswith("# ") and "title" not in sections:
            sections["title"] = line.lstrip("# ").strip()
            continue
        if line.startswith("## "):
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            heading = line.lstrip("# ").strip().lower()
            if "summary" in heading:
                current_section = "summary"
            elif "key concept" in heading:
                current_section = "key_concepts"
            elif "detail" in heading or "explanation" in heading:
                current_section = "detail"
            elif "procedure" in heading or "learned" in heading:
                current_section = "procedures"
            elif "open question" in heading:
                current_section = "open_questions"
            else:
                current_section = heading.replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def format_gemma_chat(instruction: str, output: str) -> str:
    """Format an instruction pair in Gemma chat template."""
    return (
        "<start_of_turn>user\n"
        f"{instruction}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{output}"
        "<end_of_turn>"
    )


def run_plasticity_session(
    config: Optional[PlasticityConfig] = None,
) -> PlasticityResult:
    """Run a synaptic plasticity (LoRA fine-tuning) session.

    This is the heavyweight operation — loads the model, trains,
    and saves the adapter. Only call during dreaming cycles when
    GPU 1 is free (Id inference is paused).

    Args:
        config: Plasticity configuration.

    Returns:
        PlasticityResult with training metrics.
    """
    cfg = config or PlasticityConfig()
    t0 = time.monotonic()

    # Step 1: Extract instruction pairs from wiki
    pairs = extract_instruction_pairs(cfg.wiki_dir, cfg.max_examples)

    if len(pairs) < 5:
        return PlasticityResult(
            examples_generated=len(pairs),
            examples_trained=0,
            training_loss=0.0,
            duration_seconds=time.monotonic() - t0,
            adapter_path="",
            skipped_reason=(
                f"Too few examples ({len(pairs)}). "
                "Need at least 5 high-certainty wiki articles."
            ),
        )

    # Step 2: Format for training
    train_data = [{"text": format_gemma_chat(p.instruction, p.output)} for p in pairs]

    logger.info(
        "[plasticity] %d training examples from %d articles",
        len(train_data),
        len(set(p.source_article for p in pairs)),
    )

    # Step 3: Run fine-tuning (requires GPU + unsloth)
    training_loss = 0.0
    try:
        import torch
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig

        torch.cuda.set_device(cfg.gpu_index)

        logger.info("[plasticity] Loading %s on GPU %d", cfg.model_name, cfg.gpu_index)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map={"": cfg.gpu_index},
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            max_seq_length=cfg.max_seq_length,
        )

        dataset = Dataset.from_list(train_data)
        os.makedirs(cfg.output_dir, exist_ok=True)

        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=cfg.learning_rate,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            fp16=True,
            bf16=False,
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            seed=42,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        stats = trainer.train()
        training_loss = stats.training_loss

        # Save adapter
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        logger.info(
            "[plasticity] Training complete: loss=%.4f, steps=%d",
            training_loss,
            stats.global_step,
        )

        # Clean up GPU memory
        del model, tokenizer, trainer
        torch.cuda.empty_cache()

    except ImportError as e:
        logger.warning(
            "[plasticity] Fine-tuning dependencies not available: %s. "
            "Saving training data for later.",
            e,
        )
        # Save the training data as JSONL for manual fine-tuning
        jsonl_path = os.path.join(cfg.output_dir, "dream_finetune.jsonl")
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(jsonl_path, "w") as f:
            for pair in pairs:
                json.dump(
                    {
                        "instruction": pair.instruction,
                        "output": pair.output,
                        "source": pair.source_article,
                        "grade": pair.certainty_grade,
                    },
                    f,
                )
                f.write("\n")
        logger.info(
            "[plasticity] Saved %d examples to %s",
            len(pairs),
            jsonl_path,
        )

    except Exception as e:
        logger.error("[plasticity] Training failed: %s", e)
        return PlasticityResult(
            examples_generated=len(pairs),
            examples_trained=0,
            training_loss=0.0,
            duration_seconds=time.monotonic() - t0,
            adapter_path="",
            skipped_reason=f"Training error: {e}",
        )

    elapsed = time.monotonic() - t0
    return PlasticityResult(
        examples_generated=len(pairs),
        examples_trained=len(train_data),
        training_loss=training_loss,
        duration_seconds=elapsed,
        adapter_path=cfg.output_dir,
    )

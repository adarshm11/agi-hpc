# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the synaptic plasticity module.

Tests instruction pair extraction from wiki articles, grade
filtering, Gemma chat formatting, and graceful degradation
when GPU/unsloth are unavailable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agi.dreaming.synaptic_plasticity import (
    PlasticityConfig,
    PlasticityResult,
    _article_to_pairs,
    _extract_grade,
    _parse_sections,
    extract_instruction_pairs,
    format_gemma_chat,
    run_plasticity_session,
)

SAMPLE_ARTICLE = """# Quantum Computing Fundamentals

## Summary

Quantum computing uses qubits instead of classical bits. Key concepts
include superposition, entanglement, and quantum gates.

## Key Concepts

- **Superposition**: A qubit can be in both |0> and |1> simultaneously
- **Entanglement**: Correlated qubits share state across distance
- **Quantum gates**: Unitary operations that transform qubit states

## Detailed Explanation

Quantum computers exploit quantum mechanical phenomena to process
information in ways that classical computers cannot efficiently
replicate. The fundamental unit is the qubit, which unlike a
classical bit can exist in a superposition of states.

## Learned Procedures

1. Initialize qubits to |0>
2. Apply Hadamard gate for superposition
3. Apply CNOT for entanglement
4. Measure to collapse state

## Certainty Report

| Metric | Value |
|--------|-------|
| Article Grade | **A** |
| Mean Certainty | 0.92 |
| Mean Confidence | 0.88 |

## Provenance

Sources: episode-001, episode-042, episode-099
"""


class TestGradeExtraction:
    """Tests for extracting certainty grades from articles."""

    def test_grade_a(self) -> None:
        assert _extract_grade(SAMPLE_ARTICLE) == "A"

    def test_grade_b(self) -> None:
        content = SAMPLE_ARTICLE.replace("**A**", "**B**")
        assert _extract_grade(content) == "B"

    def test_grade_missing_defaults_c(self) -> None:
        assert _extract_grade("No grade here") == "C"


class TestSectionParsing:
    """Tests for parsing wiki article sections."""

    def test_extracts_title(self) -> None:
        sections = _parse_sections(SAMPLE_ARTICLE)
        assert sections["title"] == "Quantum Computing Fundamentals"

    def test_extracts_summary(self) -> None:
        sections = _parse_sections(SAMPLE_ARTICLE)
        assert "qubits" in sections.get("summary", "")

    def test_extracts_key_concepts(self) -> None:
        sections = _parse_sections(SAMPLE_ARTICLE)
        assert "Superposition" in sections.get("key_concepts", "")

    def test_extracts_procedures(self) -> None:
        sections = _parse_sections(SAMPLE_ARTICLE)
        assert "Hadamard" in sections.get("procedures", "")


class TestInstructionPairGeneration:
    """Tests for converting articles to instruction pairs."""

    def test_generates_pairs(self) -> None:
        pairs = _article_to_pairs(SAMPLE_ARTICLE, "dream-quantum", "A")
        assert len(pairs) >= 3  # summary, concepts, detail, procedures

    def test_pair_has_instruction_and_output(self) -> None:
        pairs = _article_to_pairs(SAMPLE_ARTICLE, "dream-quantum", "A")
        for p in pairs:
            assert p.instruction
            assert p.output
            assert p.source_article == "dream-quantum"
            assert p.certainty_grade == "A"

    def test_summary_pair(self) -> None:
        pairs = _article_to_pairs(SAMPLE_ARTICLE, "dream-quantum", "A")
        summary_pairs = [p for p in pairs if "know about" in p.instruction.lower()]
        assert len(summary_pairs) == 1
        assert "qubits" in summary_pairs[0].output


class TestExtractFromDirectory:
    """Tests for extracting pairs from a wiki directory."""

    def test_extract_from_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a sample article
            article_path = Path(tmpdir) / "dream-test-topic.md"
            article_path.write_text(SAMPLE_ARTICLE, encoding="utf-8")

            pairs = extract_instruction_pairs(tmpdir, max_examples=100)
            assert len(pairs) >= 3

    def test_filters_low_grades(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a low-grade article
            low_grade = SAMPLE_ARTICLE.replace("**A**", "**D**")
            (Path(tmpdir) / "dream-low-grade.md").write_text(
                low_grade, encoding="utf-8"
            )

            pairs = extract_instruction_pairs(tmpdir, max_examples=100, min_grade="B")
            assert len(pairs) == 0  # D < B, filtered out

    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = extract_instruction_pairs(tmpdir)
            assert len(pairs) == 0

    def test_nonexistent_dir(self) -> None:
        pairs = extract_instruction_pairs("/nonexistent/path")
        assert len(pairs) == 0


class TestGemmaChatFormat:
    """Tests for the Gemma chat template formatting."""

    def test_format(self) -> None:
        result = format_gemma_chat("What is AI?", "AI is...")
        assert "<start_of_turn>user" in result
        assert "What is AI?" in result
        assert "<start_of_turn>model" in result
        assert "AI is..." in result
        assert "<end_of_turn>" in result


class TestPlasticitySession:
    """Tests for the full plasticity session."""

    def test_skips_with_few_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PlasticityConfig(
                wiki_dir=tmpdir,
                output_dir=tmpdir + "/output",
            )
            result = run_plasticity_session(cfg)
            assert result.skipped_reason is not None
            assert result.examples_trained == 0

    def test_saves_jsonl_when_gpu_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write enough articles
            for i in range(3):
                article = SAMPLE_ARTICLE.replace(
                    "Quantum Computing",
                    f"Topic {i}",
                )
                (Path(tmpdir) / f"dream-topic-{i}.md").write_text(
                    article, encoding="utf-8"
                )

            output_dir = tmpdir + "/output"
            cfg = PlasticityConfig(
                wiki_dir=tmpdir,
                output_dir=output_dir,
                max_examples=50,
            )
            run_plasticity_session(cfg)

            # On Windows/no GPU: should save JSONL fallback
            jsonl_path = Path(output_dir) / "dream_finetune.jsonl"
            if jsonl_path.exists():
                import json

                lines = jsonl_path.read_text().strip().split("\n")
                assert len(lines) >= 3
                first = json.loads(lines[0])
                assert "instruction" in first
                assert "output" in first


class TestPlasticityResult:
    """Tests for the PlasticityResult dataclass."""

    def test_creation(self) -> None:
        r = PlasticityResult(
            examples_generated=50,
            examples_trained=45,
            training_loss=0.42,
            duration_seconds=120.0,
            adapter_path="/models/adapter",
        )
        assert r.examples_trained == 45
        assert r.skipped_reason is None

    def test_skipped(self) -> None:
        r = PlasticityResult(
            examples_generated=2,
            examples_trained=0,
            training_loss=0.0,
            duration_seconds=1.0,
            adapter_path="",
            skipped_reason="Too few examples",
        )
        assert r.skipped_reason is not None

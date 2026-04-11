#!/bin/bash
## Download latest Gemma 4 chat templates from HuggingFace
## Run manually or via cron: 0 3 * * 0 /home/claude/agi-hpc/scripts/update_gemma_templates.sh
##
## These templates fix tool calling and reasoning budget issues.
## Source: https://github.com/ggml-org/llama.cpp/pull/21697

set -e
TEMPLATE_DIR="/home/claude/models/templates"
mkdir -p "$TEMPLATE_DIR"

echo "Downloading Gemma 4 chat templates..."

curl -sf -o "$TEMPLATE_DIR/gemma-4-31B-it.jinja" \
  "https://huggingface.co/google/gemma-4-31B-it/raw/main/chat_template.jinja" \
  && echo "  31B: OK" || echo "  31B: FAILED"

curl -sf -o "$TEMPLATE_DIR/gemma-4-26B-A4B-it.jinja" \
  "https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja" \
  && echo "  26B-A4B: OK" || echo "  26B-A4B: FAILED"

curl -sf -o "$TEMPLATE_DIR/gemma-4-E4B-it.jinja" \
  "https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja" \
  && echo "  E4B: OK" || echo "  E4B: FAILED"

curl -sf -o "$TEMPLATE_DIR/gemma-4-E2B-it.jinja" \
  "https://huggingface.co/google/gemma-4-E2B-it/raw/main/chat_template.jinja" \
  && echo "  E2B: OK" || echo "  E2B: FAILED"

echo "Templates saved to $TEMPLATE_DIR"
ls -la "$TEMPLATE_DIR"/gemma-4-*.jinja

echo ""
echo "To use: add --chat-template-file $TEMPLATE_DIR/gemma-4-31B-it.jinja"
echo "to the llama-server command in the systemd service file."

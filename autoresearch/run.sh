#!/bin/bash
# Quick start script for autoresearch Sudoku experiments.
#
# Usage:
#   ./run.sh              # Run one experiment (~5 min)
#   ./run.sh --prepare    # Download and preprocess data first
#
# To start the autonomous agent loop, point your AI agent at program.md:
#   "Read autoresearch/program.md and let's kick off a new experiment!"

set -e
cd "$(dirname "$0")"

if [ "$1" = "--prepare" ]; then
    echo "=== Preparing data ==="
    python prepare.py
    echo ""
    echo "Data ready. Now run: ./run.sh"
    exit 0
fi

# Check data exists
if [ ! -f "../data/processed/train_questions.npy" ]; then
    echo "Data not found. Run: ./run.sh --prepare"
    exit 1
fi

echo "=== Running experiment ==="
python train.py 2>&1 | tee run.log

echo ""
echo "=== Results ==="
grep "^puzzle_accuracy:\|^cell_accuracy:\|^peak_vram_mb:\|^training_seconds:" run.log

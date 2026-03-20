# autoresearch — Sudoku Solver

This is an experiment to have an LLM autonomously research and improve a Sudoku solver model.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - This file (`program.md`) — your instructions.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `data/processed/` contains `.npy` files. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task

You are training a neural network to solve Sudoku puzzles. The model takes a flattened 9x9 grid (81 cells) with values 0-9 (0=unknown, 1-9=given digits) and predicts the digit (1-9) for each cell. The metric is **puzzle_accuracy** — the percentage of puzzles where ALL unknown cells are correctly predicted.

## Dataset

- **Source**: `sapientinc/sudoku-extreme` from HuggingFace, filtered to easy sources.
- **Training set**: ~2M samples (with 1x augmentation via Sudoku-preserving permutations).
- **Test set**: ~115K samples (no augmentation).
- **Input**: 81 integer values (0-9), where 0 = unknown cell.
- **Target**: 81 integer values (0-8), which are the true digits minus 1 (for CrossEntropy).
- **Mask**: Boolean mask where True = cell was unknown (needs prediction).
- Loss is computed only on masked (unknown) cells.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `cd autoresearch && python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture (LSTM, Transformer, CNN, hybrid, etc.), optimizer, hyperparameters, training loop, batch size, model size, loss function, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, grid size, etc).
- Install new packages or add dependencies. You can only use what's already installed (PyTorch, numpy, standard library).
- Modify the evaluation harness. The `evaluate_puzzle_accuracy` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest puzzle_accuracy.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful puzzle_accuracy gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 puzzle_accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Ideas to explore

Here are some directions you might try (but you are not limited to these):

### Architecture
- **Transformer encoder**: Self-attention can naturally capture row/column/box constraints. Add positional embeddings for row, column, and 3x3 box indices.
- **Hybrid LSTM + Attention**: Use LSTM for sequential processing with cross-attention for constraint awareness.
- **Deeper/wider LSTM**: Try different hidden sizes (256, 512, 768, 1024) and layer counts (4, 6, 8, 12).
- **Residual connections**: Add skip connections between LSTM layers.
- **Multi-pass / iterative refinement**: Run the model multiple times, feeding predictions back as input.

### Training
- **Learning rate**: Try different LRs (1e-4 to 1e-2), different schedulers (cosine, OneCycleLR, warmup+decay).
- **Optimizer**: Try AdamW with weight decay, or different beta values.
- **Label smoothing**: Small amount (0.05-0.1) may help generalization.
- **Gradient accumulation**: If batch size is limited by VRAM, accumulate gradients.
- **Curriculum learning**: Start with easier puzzles (more given cells) and progress to harder ones.

### Input representation
- **Constraint embeddings**: Add row/column/box position embeddings (like the Transformer model does).
- **One-hot vs embedding**: The current model uses learned embeddings; experiment with the dimension.
- **Board structure**: Reshape as 9x9 grid and use 2D operations.

### Loss
- **Auxiliary losses**: Add constraint-checking losses (no duplicate in row/col/box).
- **Focal loss**: Focus on harder cells.
- **Different masking strategies**: Also predict known cells during training.

## Output format

Once the script finishes it prints a summary like this:

```
---
puzzle_accuracy:  0.123456
cell_accuracy:    0.987654
avg_loss:         0.543210
puzzles_solved:   14142/114558
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4500.2
num_steps:        953
num_epochs:       3
num_params:       33,620,489
hidden_size:      512
num_layers:       6
batch_size:       1024
learning_rate:    0.001
```

You can extract the key metric from the log file:

```
grep "^puzzle_accuracy:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	puzzle_accuracy	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. puzzle_accuracy achieved (e.g. 0.123456) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 4.5 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	puzzle_accuracy	memory_gb	status	description
a1b2c3d	0.123456	4.4	keep	baseline
b2c3d4e	0.156789	4.5	keep	increase hidden size to 768
c3d4e5f	0.110000	4.3	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `cd /home/user/lstm_sudoku_solver/autoresearch && python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^puzzle_accuracy:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If puzzle_accuracy improved (higher is better), you "advance" the branch, keeping the git commit
9. If puzzle_accuracy is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

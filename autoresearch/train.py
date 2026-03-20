"""
Autoresearch Sudoku solver training script. Single-GPU, single-file.
This is the ONLY file the agent modifies.

Usage: python train.py

The agent can change anything here: model architecture, optimizer,
hyperparameters, training loop, batch size, etc. Everything is fair game.
"""

import os
import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, GRID_SIZE, NUM_CLASSES, NUM_INPUT_TOKENS,
    make_dataloaders, evaluate_puzzle_accuracy,
)

# ---------------------------------------------------------------------------
# Model Architecture: Bidirectional LSTM with Embeddings
# ---------------------------------------------------------------------------

class SudokuLSTM(nn.Module):
    """Bidirectional LSTM Sudoku solver with learned embeddings.

    Input: (batch, 81) integer indices 0-9
    Output: (batch, 81, 9) logits for digits 1-9
    """

    def __init__(
        self,
        hidden_size=512,
        num_layers=6,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(NUM_INPUT_TOKENS, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def masked_loss(preds, targets, mask):
    """Compute CrossEntropyLoss only on masked (unknown) cells."""
    loss = F.cross_entropy(preds.reshape(-1, 9), targets.reshape(-1), reduction='none')
    loss = loss.reshape(targets.shape)
    masked = loss * mask.float()
    return masked.sum() / (mask.sum() + 1e-6)


def compute_accuracy(predictions, targets, mask):
    """Compute cell-level and puzzle-level accuracy."""
    predicted_classes = predictions.argmax(dim=-1)
    correct = (predicted_classes == targets) & mask
    cell_accuracy = correct.sum().float() / (mask.sum().float() + 1e-6)

    correct_per_puzzle = correct.sum(dim=1)
    masked_per_puzzle = mask.sum(dim=1)
    puzzles_solved = (correct_per_puzzle == masked_per_puzzle).float()
    puzzle_accuracy = puzzles_solved.mean()

    return cell_accuracy.item(), puzzle_accuracy.item()


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_SIZE = 512           # LSTM hidden size
NUM_LAYERS = 6              # number of stacked LSTM layers
DROPOUT = 0.3               # dropout between LSTM layers

# Optimization
BATCH_SIZE = 1024           # training batch size
LEARNING_RATE = 1e-3        # initial learning rate
WEIGHT_DECAY = 0.0          # AdamW weight decay
MAX_GRAD_NORM = 1.0         # gradient clipping max norm
LABEL_SMOOTHING = 0.0       # label smoothing factor

# LR Schedule
USE_ONECYCLE = False        # use OneCycleLR (True) or ReduceLROnPlateau (False)
WARMUP_RATIO = 0.1          # fraction of steps for warmup (OneCycleLR only)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Time budget: {TIME_BUDGET}s")

# Data
train_loader, test_loader = make_dataloaders(BATCH_SIZE)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# Model
model = SudokuLSTM(
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

# Compile model if available
if hasattr(torch, 'compile') and device.type == 'cuda':
    model = torch.compile(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# LR Scheduler
if USE_ONECYCLE:
    # Estimate total steps from time budget (rough: assume ~0.5s per batch)
    estimated_steps = max(len(train_loader) * 20, 1000)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=estimated_steps,
        pct_start=WARMUP_RATIO,
        anneal_strategy='cos',
    )
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )

# Mixed precision
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device.type == 'cuda' else torch.amp.autocast(device_type="cpu", enabled=False)

# ---------------------------------------------------------------------------
# Training Loop (time-budgeted)
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("Starting training...")
print(f"{'='*60}\n")

t_start_training = time.time()
total_training_time = 0.0
step = 0
epoch = 0
best_train_puzzle_acc = 0.0
smooth_loss = 0.0

# GC management
gc.collect()
gc.freeze()
gc.disable()

while True:
    model.train()
    epoch += 1
    epoch_loss = 0.0
    epoch_cell_acc = 0.0
    epoch_puzzle_acc = 0.0
    epoch_batches = 0

    for batch in train_loader:
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()

        q = batch['question'].to(device, non_blocking=True)
        a = batch['answer'].to(device, non_blocking=True)
        m = batch['mask'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            preds = model(q)
            if LABEL_SMOOTHING > 0:
                loss = F.cross_entropy(
                    preds.reshape(-1, 9), a.reshape(-1),
                    reduction='none', label_smoothing=LABEL_SMOOTHING,
                )
                loss = loss.reshape(a.shape)
                loss = (loss * m.float()).sum() / (m.sum() + 1e-6)
            else:
                loss = masked_loss(preds, a, m)

        if scaler is not None:
            scaler.scale(loss).backward()
            if MAX_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if USE_ONECYCLE:
            scheduler.step()

        # Track metrics
        with torch.no_grad():
            cell_acc, puzzle_acc = compute_accuracy(preds, a, m)

        loss_val = loss.item()
        epoch_loss += loss_val
        epoch_cell_acc += cell_acc
        epoch_puzzle_acc += puzzle_acc
        epoch_batches += 1
        step += 1

        # Smoothed loss for logging
        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased_loss = smooth_loss / (1 - ema_beta ** step)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        dt = time.time() - t0

        # Only count time after first few warmup steps (compilation)
        if step > 5:
            total_training_time += dt

        remaining = max(0, TIME_BUDGET - total_training_time)

        # Log every 50 steps
        if step % 50 == 0:
            pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
            print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | "
                  f"cell: {cell_acc:.2%} | puzzle: {puzzle_acc:.2%} | "
                  f"dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ",
                  end="", flush=True)

        # Fast fail
        if math.isnan(loss_val) or loss_val > 100:
            print("\nFAIL: loss exploded")
            exit(1)

        # Time's up
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

        # GC every 2000 steps
        if step % 2000 == 0:
            gc.collect()

    # End of epoch stats
    avg_loss = epoch_loss / max(epoch_batches, 1)
    avg_cell = epoch_cell_acc / max(epoch_batches, 1)
    avg_puzzle = epoch_puzzle_acc / max(epoch_batches, 1)

    if not USE_ONECYCLE:
        scheduler.step(avg_loss)

    best_train_puzzle_acc = max(best_train_puzzle_acc, avg_puzzle)

    print(f"\nEpoch {epoch} done | loss: {avg_loss:.4f} | "
          f"cell: {avg_cell:.2%} | puzzle: {avg_puzzle:.2%}")

    # Time's up
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print(f"\n{'='*60}")
print("Training complete. Running evaluation...")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Final Evaluation
# ---------------------------------------------------------------------------

model.eval()
with autocast_ctx:
    results = evaluate_puzzle_accuracy(model, device=device)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == 'cuda' else 0

# ---------------------------------------------------------------------------
# Final Summary (parsed by the experiment loop)
# ---------------------------------------------------------------------------

print("---")
print(f"puzzle_accuracy:  {results['puzzle_accuracy']:.6f}")
print(f"cell_accuracy:    {results['cell_accuracy']:.6f}")
print(f"avg_loss:         {results['avg_loss']:.6f}")
print(f"puzzles_solved:   {results['puzzles_solved']}/{results['total_puzzles']}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {epoch}")
print(f"num_params:       {num_params:,}")
print(f"hidden_size:      {HIDDEN_SIZE}")
print(f"num_layers:       {NUM_LAYERS}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"learning_rate:    {LEARNING_RATE}")

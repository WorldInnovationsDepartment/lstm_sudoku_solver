"""
Fixed data preparation and evaluation for autoresearch experiments.
Downloads data, preprocesses it, and provides runtime utilities.

Usage:
    python prepare.py                  # full prep (download + preprocess)
    python prepare.py --num-aug 2      # use 2x augmentation factor

Data is stored in data/processed/ relative to the repo root.

DO NOT MODIFY THIS FILE. It contains the fixed evaluation, data loading,
and training constants. The agent only modifies train.py.
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
GRID_SIZE = 81             # 9x9 Sudoku grid
NUM_CLASSES = 9            # digits 1-9
NUM_INPUT_TOKENS = 10      # 0 (unknown) + digits 1-9
EVAL_BATCH_SIZE = 1024     # batch size for evaluation

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")

# Dataset source
EASY_SOURCES = ['puzzles0_kaggle', 'puzzles1_unbiased', 'puzzles2_17_clue']

# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

def shuffle_sudoku(board_flat, solution_flat):
    """Apply valid Sudoku transformations to a board and solution."""
    board = board_flat.reshape(9, 9)
    sol = solution_flat.reshape(9, 9)

    # 1. Permute digits (1-9)
    digit_map = np.arange(10)
    digit_map[1:] = np.random.permutation(np.arange(1, 10))

    # 2. Random Transpose
    if np.random.rand() < 0.5:
        board = board.T
        sol = sol.T

    # 3. Permute Bands (groups of 3 rows) + rows within bands
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # 4. Permute Stacks (groups of 3 cols) + cols within stacks
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    board = board[row_perm, :][:, col_perm]
    sol = sol[row_perm, :][:, col_perm]

    board = digit_map[board]
    sol = digit_map[sol]

    return board.flatten(), sol.flatten()


# ---------------------------------------------------------------------------
# Data Download & Preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataset(output_dir=DATA_DIR, num_aug=1):
    """Download, filter, augment, and save dataset as .npy."""
    if os.path.exists(os.path.join(output_dir, "train_questions.npy")):
        print(f"Dataset already exists at {output_dir}. Skipping generation.")
        return

    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading dataset from HuggingFace...")
    ds = load_dataset("sapientinc/sudoku-extreme")

    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'test']:
        print(f"Processing {split} split...")
        split_ds = ds[split].filter(lambda x: x['source'] in EASY_SOURCES)

        questions = []
        answers = []

        print("Converting to integers and augmenting...")
        for item in tqdm(split_ds):
            q = np.array([0 if c == '.' else int(c) for c in item['question']], dtype=np.uint8)
            a = np.array([int(c) for c in item['answer']], dtype=np.uint8)

            questions.append(q)
            answers.append(a)

            if split == 'train' and num_aug > 0:
                for _ in range(num_aug):
                    q_aug, a_aug = shuffle_sudoku(q, a)
                    questions.append(q_aug.astype(np.uint8))
                    answers.append(a_aug.astype(np.uint8))

        q_arr = np.array(questions, dtype=np.uint8)
        a_arr = np.array(answers, dtype=np.uint8)

        print(f"Saving {len(q_arr)} samples to {output_dir}...")
        np.save(os.path.join(output_dir, f"{split}_questions.npy"), q_arr)
        np.save(os.path.join(output_dir, f"{split}_answers.npy"), a_arr)

    print("Data preprocessing complete.")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class SudokuDataset(Dataset):
    """Fast Sudoku dataset loading from preprocessed .npy files."""

    def __init__(self, data_dir, split):
        self.questions = np.load(os.path.join(data_dir, f"{split}_questions.npy"))
        self.answers = np.load(os.path.join(data_dir, f"{split}_answers.npy"))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx].astype(np.int64)
        a = self.answers[idx].astype(np.int64)
        mask = (q == 0)
        target = a - 1  # shift from 1-9 to 0-8 for CrossEntropy
        return {
            'question': torch.from_numpy(q),
            'answer': torch.from_numpy(target),
            'mask': torch.from_numpy(mask),
        }


def make_dataloaders(batch_size, data_dir=DATA_DIR, num_workers=2):
    """Create train and test dataloaders."""
    train_ds = SudokuDataset(data_dir, "train")
    test_ds = SudokuDataset(data_dir, "test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_puzzle_accuracy(model, data_dir=DATA_DIR, device="cuda"):
    """
    Evaluate puzzle-level accuracy on the test set.
    This is the ground truth metric: percentage of puzzles completely solved.

    Returns dict with cell_accuracy, puzzle_accuracy, and avg_loss.
    """
    test_ds = SudokuDataset(data_dir, "test")
    test_loader = DataLoader(
        test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model.eval()
    total_correct_cells = 0
    total_masked_cells = 0
    total_puzzles_solved = 0
    total_puzzles = 0
    total_loss = 0.0
    num_batches = 0

    for batch in test_loader:
        q = batch['question'].to(device)
        a = batch['answer'].to(device)
        m = batch['mask'].to(device)

        preds = model(q)

        # Loss (no label smoothing for eval)
        loss = F.cross_entropy(preds.reshape(-1, 9), a.reshape(-1), reduction='none')
        loss = loss.reshape(a.shape)
        masked_loss = (loss * m.float()).sum() / (m.sum() + 1e-6)
        total_loss += masked_loss.item()

        # Accuracy
        predicted = preds.argmax(dim=-1)
        correct = (predicted == a) & m
        total_correct_cells += correct.sum().item()
        total_masked_cells += m.sum().item()

        correct_per_puzzle = correct.sum(dim=1)
        masked_per_puzzle = m.sum(dim=1)
        total_puzzles_solved += (correct_per_puzzle == masked_per_puzzle).sum().item()
        total_puzzles += q.size(0)
        num_batches += 1

    cell_acc = total_correct_cells / max(total_masked_cells, 1)
    puzzle_acc = total_puzzles_solved / max(total_puzzles, 1)
    avg_loss = total_loss / max(num_batches, 1)

    return {
        'cell_accuracy': cell_acc,
        'puzzle_accuracy': puzzle_acc,
        'avg_loss': avg_loss,
        'puzzles_solved': total_puzzles_solved,
        'total_puzzles': total_puzzles,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch Sudoku experiments")
    parser.add_argument("--num-aug", type=int, default=1,
                        help="Number of augmentations per training sample (0=none)")
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    print()

    preprocess_dataset(num_aug=args.num_aug)
    print()
    print("Done! Ready to train.")

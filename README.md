# Sudoku Solver with LSTM (just was curious to compare bidirectional LSTM with HRM ~ params count)

A deep learning approach to solving Sudoku puzzles using a bidirectional LSTM neural network.

## Overview

This project trains an LSTM-based model to solve Sudoku puzzles by treating the 81-cell grid as a sequence prediction problem. The model learns to fill in unknown cells while respecting Sudoku constraints.

### Key Features

- **Bidirectional LSTM Architecture**: Captures constraints from both directions in the sequence
- **Masked Loss Function**: Only computes loss on unknown cells, allowing the model to focus on predictions
- **Large-Scale Training**: Trained on over 1 million puzzles from the HuggingFace dataset
- **Multiple Accuracy Metrics**: Tracks both cell-level and puzzle-level accuracy

## Model Architecture

```
Input (81, 10) → Bidirectional LSTM (6 layers, 512 hidden) → FC Layer → Output (81, 9)
```

- **Input**: 81 cells, each one-hot encoded with 10 dimensions (0 = unknown, 1-9 = digits)
- **LSTM**: 6-layer bidirectional LSTM with 512 hidden units and 0.3 dropout
- **Output**: 9-class prediction for each cell (digits 1-9)
- **Total Parameters**: ~33.6M

## Dataset

Uses the [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) dataset from HuggingFace.

The training is filtered to following puzzle sources:
- `puzzles0_kaggle`
- `puzzles1_unbiased`
- `puzzles2_17_clue`

| Split | Samples |
|-------|---------|
| Train | 1,034,600 |
| Test  | 114,558 |

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sudoku_solver
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Login to HuggingFace (required to access the dataset):
```bash
huggingface-cli login
```

## Usage

### Training

Open and run the Jupyter notebook:
```bash
jupyter notebook train.ipynb
```

The notebook contains:
1. Dataset loading and preprocessing
2. Model definition (`SudokuLSTM`)
3. Training loop with masked cross-entropy loss
4. Evaluation and visualization

### Inference

After training, use the `solve_sudoku()` function to solve new puzzles:

```python
puzzle = '..8..23412........9......5.........4..3..89.7....53....6.3.1.7...7.4.8...5.8.9...'
solution, confidences = solve_sudoku(model, puzzle, device)
```

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 128 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau |
| Epochs | 2 |

## Project Structure

```
sudoku_solver/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── train.ipynb         # Training notebook
└── best_sudoku_model.pt  # Saved model weights (after training)
```

## Requirements

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended) or Apple Silicon (MPS)

See `requirements.txt` for full dependencies.

## How It Works

1. **Data Representation**: Each Sudoku puzzle is represented as a string of 81 characters where `.` represents unknown cells and digits 1-9 represent known values.

2. **One-Hot Encoding**: Input cells are one-hot encoded with 10 dimensions (index 0 for unknown, indices 1-9 for known digits).

3. **Masked Training**: Loss is computed only on cells that were originally unknown, allowing the model to focus on prediction rather than memorization of given values.

4. **Bidirectional Processing**: The LSTM processes the sequence in both directions, helping capture Sudoku constraints that depend on cells in any direction.

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme)
- Built with PyTorch and HuggingFace Datasets

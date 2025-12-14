# Sudoku Training Improvements

This document details the optimizations applied to `train_improved.ipynb` to achieve significantly faster training and higher throughput, inspired by the HRM (Hierarchical Reasoning Model) pipeline.

## Key Improvements

### 1. Offline Data Preprocessing & Binary Format
**Justification**: The original pipeline parsed string-based Sudoku puzzles (e.g., `"12..4..."`) inside the `Dataset.__getitem__` method. This caused the CPU to be a bottleneck, as it had to parse 81 characters for every single sample in every epoch.
**Improvement**:
-   We now preprocess the dataset once and save it as INT8 binary files (`.npy`).
-   Data loading is now instant (limited only by memory bandwidth), removing the CPU bottleneck.
-   Storage is efficient (Integers take less space than strings).

### 2. Data Augmentation
**Justification**: To train for 20,000+ epochs without overfitting, we need a massive effective dataset size.
**Improvement**:
-   Implemented Sudoku-preserving augmentations:
    -   Permutation of digits (1-9).
    -   Permutation of bands (blocks of 3 rows).
    -   Permutation of rows within bands.
    -   Permutation of stacks (blocks of 3 columns).
    -   Permutation of columns within stacks.
    -   Transposition.
-   These are applied during the preprocessing stage to expand the dataset size effectively.

### 3. Learnable Embeddings instead of One-Hot Encoding
**Justification**: The original model used One-Hot encoding for inputs `(Batch, 81, 10)`. This increases memory usage and computation.
**Improvement**:
-   Switched to `nn.Embedding(10, hidden_size)`.
-   Input is now `(Batch, 81)` (integer indices).
-   This significantly reduces memory footprint, allowing for much larger batch sizes (e.g., 1024 vs 128).

### 4. Mixed Precision Training (AMP)
**Justification**: Modern GPUs (and Apple Silicon) support half-precision floating point (`float16` or `bfloat16`), which is faster and uses less memory.
**Improvement**:
-   Added `torch.amp.autocast` and `GradScaler`.
-   This typically yields a 1.5x - 2x speedup on compatible hardware.

### 5. Increased Batch Size
**Justification**: With the memory savings from Embeddings and efficient data loading, we can saturate the GPU better.
**Improvement**:
-   Increased default batch size from 128 to 1024.
-   This utilizes GPU parallelism more effectively, drastically increasing `samples/second`.

## How to Run

1.  Open `train_improved.ipynb`.
2.  Run the **Preprocessing** cells first. This will download the dataset and create a `data/processed` directory.
3.  Run the **Training** cells.

## Performance Comparison (Estimated)

| Metric | Original | Improved |
| :--- | :--- | :--- |
| **Data Format** | Strings (Parsed on-the-fly) | Binary `.npy` (Pre-computed) |
| **Input** | One-Hot (81x10 floats) | Indices (81 ints) + Embedding |
| **Batch Size** | 128 | 1024+ |
| **Throughput** | ~2k samples/sec | ~50k+ samples/sec (expected) |

# MuloAD: A Multi-Omics Integration Model Utilizing Graph Convolutional Networks for Alzheimer's Disease Diagnosis and Biomarker Identification

## Features

- Integration of multiple omics data types (e.g., genomics, transcriptomics, proteomics)
- Graph-based representation learning for each omics data type
- View-correlation discovery network for effective multi-omics fusion
- Support for binary ands multi-class classification tasks
- Performance evaluation with accuracy, F1-score, and AUC metrics

## Requirements

- Python 3.12
- PyTorch
- NumPy
- scikit-lear
- Matplotlib


## Usage

### Data Format

Prepare your multi-omics data in CSV format:
- `{view_id}_tr.csv`: Training data for each view
- `{view_id}_te.csv`: Testing data for each view
- `labels_tr.csv`: Training labels
- `labels_te.csv`: Testing labels

## Example

See `main.py` for a complete example of how to use MuloAD with the ROSMAP dataset.


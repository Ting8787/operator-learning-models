# Operator Learning Models

This repository contains PyTorch implementations and experiments for several operator learning models.

We compare several neural-operator architectures on two benchmark problems:

1. A one-dimensional kernel integral operator
2. A two-dimensional Poisson solution operator on a disk

Because these two problems have different structures, we use different model sets for the two tasks. For the one-dimensional kernel operator, the implemented models are DNN, softmax attention, linear attention, DeepONet, and FNO. For the two-dimensional Poisson problem, the implemented models are Transformer, DeepONet, and FNO.

## How to run the code

Most experiments were developed and tested on Google Colab.

The notebooks can be opened directly in Colab. 

### One-dimensional kernel operator

| Model | Notebook |
|---|---|
| DNN | [Open in Colab](https://colab.research.google.com/drive/1G9vUJZQEJa00vf0GTqfjXTgq5gQtjx0A?usp=sharing) |
| Softmax Attention and Linear Attention | [Open in Colab](https://colab.research.google.com/drive/1U5tPypzrGBEoZCMXR6NeSVWQ-IKeIwHD?usp=sharing) |
| DeepONet | [Open in Colab](https://colab.research.google.com/drive/1RqWtK4qk11lr06k4HPeuDCsaxbDstD-a?usp=sharing) |
| FNO-1D | [Open in Colab](https://colab.research.google.com/drive/1OiZHsZmNt2qxOT74qO4HPOY2fnaJ4WLC?usp=sharing) |

### Two-dimensional Poisson problem

| Model | Notebook |
|---|---|
| Transformer | [Open in Colab](你的 Transformer Colab 連結) |
| DeepONet | [Open in Colab](你的 Poisson DeepONet Colab 連結) |
| FNO-2D | [Open in Colab](你的 Poisson FNO Colab 連結) |

```markdown
## Data

The experiments use pre-generated dataset files that are loaded directly in the notebooks.

For the one-dimensional kernel operator, the dataset is saved as:

```text
integral_gaussian_L64_N3000_seed20250930.pt

For the two-dimensional Poisson problem, the dataset is saved as:
```text
circular_poisson_dataset.npz
circular_poisson_dataset_params.pkl

## Reproducibility

Random seeds and train/validation/test splits are fixed in the experiments whenever possible.
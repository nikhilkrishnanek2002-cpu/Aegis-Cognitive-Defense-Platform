# Enhanced Evaluation Pipeline - Comprehensive Guide

## Overview

The **Enhanced Evaluation Pipeline** (`src/evaluation_enhanced.py`) provides production-grade evaluation capabilities for multi-class radar AI classification models. It computes comprehensive metrics, saves results to JSON for reproducibility, and returns structured dictionaries for downstream visualization and analysis.

### Key Features

✅ **Comprehensive Metrics**
- Overall accuracy
- Per-class precision, recall, F1 scores
- Macro-averaged metrics (unweighted by class imbalance)
- Weighted-averaged metrics (account for class distribution)
- Confusion matrix (raw and normalized)
- ROC-AUC scores (binary and multi-class One-vs-Rest)

✅ **Persistent Storage**
- Metrics saved as JSON to `outputs/reports/metrics.json`
- Full reproducibility with timestamps and configuration metadata
- Classification report for detailed per-class analysis

✅ **Structured Output**
- Returns dictionary with all metrics
- Compatible with plotting modules (reporting module)
- Ready for downstream analysis and comparison

✅ **PyTorch Integration**
- Direct evaluation of PyTorch models on dataloaders
- Automatic probability extraction and normalization
- Support for both single-input and multi-input models

---

## Installation & Setup

### Dependencies

```bash
pip install numpy scikit-learn
```

The module requires:
- `numpy` — Numerical computations
- `scikit-learn` — Metrics computation
- `torch` (optional) — For PyTorch integration

### Quick Start

```python
from src.evaluation_enhanced import compute_comprehensive_metrics

# After model evaluation
metrics = compute_comprehensive_metrics(
    predictions=y_pred,      # Array of predicted class labels
    labels=y_true,           # Array of ground truth labels
    probabilities=y_probs,   # Optional: (n_samples, n_classes) probability matrix
    output_dir="outputs/reports",
    model_name="my_radar_model"
)

# Metrics automatically saved to outputs/reports/metrics.json
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_avg']['f1']:.4f}")
```

---

## API Reference

### Main Function: `compute_comprehensive_metrics()`

Compute all evaluation metrics for multi-class classification.

#### Signature

```python
def compute_comprehensive_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    output_dir: str = "outputs/reports",
    model_name: str = "radar_model",
    num_classes: Optional[int] = None,
) -> Dict[str, Any]
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `predictions` | `np.ndarray` | — | Predicted class labels, shape `(n_samples,)`, dtype `int` |
| `labels` | `np.ndarray` | — | Ground truth labels, shape `(n_samples,)`, dtype `int` |
| `probabilities` | `Optional[np.ndarray]` | `None` | Predicted probabilities, shape `(n_samples, n_classes)`, dtype `float`. If provided, used for ROC-AUC computation |
| `output_dir` | `str` | `"outputs/reports"` | Directory to save `metrics.json` |
| `model_name` | `str` | `"radar_model"` | Model identifier for logging |
| `num_classes` | `Optional[int]` | `None` | Number of classes. If `None`, inferred from data |

#### Returns

Dictionary with structure:

```python
{
    'accuracy': float,
    'macro_avg': {
        'precision': float,
        'recall': float,
        'f1': float,
    },
    'weighted_avg': {
        'precision': float,
        'recall': float,
        'f1': float,
    },
    'per_class': {
        'precision': List[float],  # Per-class precision
        'recall': List[float],      # Per-class recall
        'f1': List[float],          # Per-class F1 score
    },
    'cm': List[List[int]],          # Confusion matrix
    'cm_normalized': List[List[float]],  # Normalized confusion matrix (row-wise)
    'roc_auc': Optional[float],     # Binary classification only
    'roc_auc_macro': Optional[float],    # Multi-class One-vs-Rest macro
    'roc_auc_weighted': Optional[float], # Multi-class One-vs-Rest weighted
    'classification_report': Dict,  # Full sklearn classification report
    'metadata': {
        'n_samples': int,
        'n_classes': int,
        'model_name': str,
        'timestamp': str,
    },
    '_metrics_file': str,  # Path to saved JSON file
}
```

#### Example

```python
import numpy as np
from src.evaluation_enhanced import compute_comprehensive_metrics

# Generate synthetic data (3-class problem)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 0])

# Optional: probabilities for ROC-AUC
y_probs = np.random.rand(10, 3)
y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

# Compute metrics
metrics = compute_comprehensive_metrics(
    predictions=y_pred,
    labels=y_true,
    probabilities=y_probs,
    output_dir="outputs/reports",
    model_name="radar_classifier"
)

# Access results
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Class 0 F1: {metrics['per_class']['f1'][0]:.4f}")
print(f"Confusion matrix:\n{np.array(metrics['cm'])}")
```

---

### PyTorch Integration: `evaluate_pytorch_enhanced()`

Directly evaluate PyTorch models on dataloaders.

#### Signature

```python
def evaluate_pytorch_enhanced(
    model,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    output_dir: str = "outputs/reports",
    model_name: str = "radar_model",
) -> Dict[str, Any]
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | PyTorch model |
| `loader` | `DataLoader` | — | DataLoader yielding batches. Format: `(inputs, labels)` or `(input1, input2, ..., labels)` |
| `device` | `str` | `"cpu"` | Device to run model on (`"cpu"`, `"cuda"`, etc.) |
| `output_dir` | `str` | `"outputs/reports"` | Output directory for metrics.json |
| `model_name` | `str` | `"radar_model"` | Model identifier |

#### Returns

Same as `compute_comprehensive_metrics()`

#### Example

```python
import torch
from src.evaluation_enhanced import evaluate_pytorch_enhanced

# After training, evaluate on test set
metrics = evaluate_pytorch_enhanced(
    model=model,
    loader=test_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="outputs/reports",
    model_name="radar_model_v1"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
```

---

### Utility Functions

#### `get_metrics_summary(metrics: Dict) -> str`

Generate human-readable summary of metrics.

```python
from src.evaluation_enhanced import get_metrics_summary

summary = get_metrics_summary(metrics)
print(summary)
```

Output:
```
======================================================================
EVALUATION METRICS SUMMARY
======================================================================

Dataset: 100 samples, 3 classes
Model: radar_classifier
Timestamp: 2026-02-20T10:11:20.448640

Overall Accuracy: 0.8500

Macro-Averaged Metrics (unweighted):
  Precision: 0.8506
  Recall:    0.8501
  F1 Score:  0.8500

...
```

#### `compare_metrics(metrics1, metrics2, model_names) -> str`

Compare two models' metrics.

```python
from src.evaluation_enhanced import compare_metrics

comparison = compare_metrics(
    metrics1=baseline_metrics,
    metrics2=improved_metrics,
    model_names=("Baseline", "Improved")
)
print(comparison)
```

#### `load_metrics_from_file(metrics_file: str) -> Dict`

Load previously saved metrics.

```python
from src.evaluation_enhanced import load_metrics_from_file

metrics = load_metrics_from_file("outputs/reports/metrics.json")
```

---

## Metrics Explained

### Accuracy
- Overall correctness: `(True Positives + True Negatives) / Total_Samples`
- **Limitation**: Misleading with class imbalance

### Precision (Per-Class)
- For class i: `TP_i / (TP_i + FP_i)`
- "Of the samples predicted as class i, how many were correct?"
- **Use case**: When false positives are costly

### Recall (Per-Class)
- For class i: `TP_i / (TP_i + FN_i)`
- "Of the actual class i samples, how many did we find?"
- **Use case**: When false negatives are costly

### F1 Score
- Harmonic mean of precision and recall: `2 * (Precision * Recall) / (Precision + Recall)`
- **Use case**: Balanced metric for imbalanced datasets

### Macro-Averaged Metrics
- Simple average across all classes, ignoring class distribution
- **Use case**: When all classes equally important

### Weighted-Averaged Metrics
- Average weighted by class support (frequency)
- **Use case**: When reflecting overall model performance

### Confusion Matrix
- `cm[i,j]` = number of samples from class i predicted as class j
- Diagonal = correct predictions
- Off-diagonal = misclassifications

### ROC-AUC
- **Binary classification**: Area under the ROC curve (Receiver Operating Characteristic)
- **Multi-class**: One-vs-Rest (OvR) approach
  - Macro: Average AUC across all OvR problems
  - Weighted: AUC weighted by class support

---

## JSON Output Format

Metrics are saved to `outputs/reports/metrics.json` with complete structure:

```json
{
  "accuracy": 0.85,
  "macro_avg": {
    "precision": 0.8506,
    "recall": 0.8501,
    "f1": 0.8500
  },
  "weighted_avg": {
    "precision": 0.8518,
    "recall": 0.85,
    "f1": 0.8505
  },
  "per_class": {
    "precision": [0.9091, 0.7941, 0.8485],
    "recall": [0.8571, 0.8182, 0.875],
    "f1": [0.8824, 0.806, 0.8615]
  },
  "cm": [[30, 2, 3], [2, 28, 5], [1, 4, 25]],
  "cm_normalized": [[0.882, 0.059, 0.059], [0.061, 0.848, 0.152], [0.038, 0.154, 0.808]],
  "roc_auc": null,
  "roc_auc_macro": 0.9017,
  "roc_auc_weighted": 0.902,
  "classification_report": {...},
  "metadata": {
    "n_samples": 100,
    "n_classes": 3,
    "model_name": "radar_classifier",
    "timestamp": "2026-02-20T10:11:20.448640"
  }
}
```

---

## Integration Workflows

### Workflow 1: Complete Evaluation Pipeline

```python
# 1. Train model
from src.train_pytorch import train_pytorch_model

model, history = train_pytorch_model(
    epochs=20,
    seed=42,
    output_dir="results"
)

# 2. Evaluate on test set
from src.evaluation_enhanced import evaluate_pytorch_enhanced, get_metrics_summary

test_loader = ...  # Your test data loader
metrics = evaluate_pytorch_enhanced(
    model=model,
    loader=test_loader,
    output_dir="outputs/reports",
    model_name="radar_model_v1"
)

# 3. Print summary
print(get_metrics_summary(metrics))

# 4. Visualize (if reporting module available)
from src.reporting import plot_confusion_matrix

plot_confusion_matrix(
    cm=np.array(metrics["cm"]),
    class_names=["Class 0", "Class 1", "Class 2"]
)
```

### Workflow 2: Model Comparison

```python
from src.evaluation_enhanced import (
    compare_metrics,
    load_metrics_from_file,
)

# Load previous metrics
old_metrics = load_metrics_from_file("outputs/reports/baseline_metrics.json")

# Evaluate new model
new_metrics = evaluate_pytorch_enhanced(model, test_loader)

# Compare
print(compare_metrics(old_metrics, new_metrics, ("Baseline", "Improved")))
```

### Workflow 3: Custom Analysis

```python
import numpy as np

# Identify problematic classes
f1_per_class = metrics["per_class"]["f1"]
worst_class = np.argmin(f1_per_class)
print(f"Worst performing: Class {worst_class} (F1={f1_per_class[worst_class]:.4f})")

# Analyze confusion
cm = np.array(metrics["cm"])
misclassifications = cm - np.diag(np.diag(cm))
confusion_pairs = np.argwhere(misclassifications > 0)
print(f"Top confusions: {confusion_pairs}")

# Class imbalance analysis
class_support = np.sum(cm, axis=1)  # Samples per class
print(f"Class distribution: {class_support}")
```

---

## Performance Considerations

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Metrics computation | O(n_samples × n_classes²) | Fast for typical problems (<1s) |
| JSON export | O(n_samples) | Negligible overhead |
| ROC-AUC computation | O(n_samples × log n_samples) | Slowest component; can skip if not needed |
| PyTorch evaluation | Depends on model | Bounded by model inference + metric computation |

For large datasets (>1M samples):
- Metrics computation: still <5 seconds
- Consider batching probabilities to manage memory

---

## Troubleshooting

### Issue: "Object of type int64 is not JSON serializable"
**Solution**: Module automatically converts numpy types. This is handled internally.

### Issue: "probabilities must have shape (n_samples, n_classes)"
**Solution**: Ensure probability matrix matches predictions shape:
```python
assert probabilities.shape[0] == len(predictions)
assert probabilities.shape[1] == num_classes
```

### Issue: "ROC-AUC computation failed"
**Solution**: ROC-AUC skipped silently if probabilities are malformed. Ensure:
```python
# Probabilities should be normalized
assert np.allclose(probabilities.sum(axis=1), 1.0)
```

### Issue: Output directory not created
**Solution**: Module auto-creates directories. If permission denied:
```python
import os
os.makedirs("outputs/reports", exist_ok=True)
```

---

## Examples

### Example 1: Binary Classification

```python
import numpy as np
from src.evaluation_enhanced import compute_comprehensive_metrics, get_metrics_summary

# Binary classification
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])
y_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.2, 0.8]])

metrics = compute_comprehensive_metrics(
    predictions=y_pred,
    labels=y_true,
    probabilities=y_probs,
    model_name="binary_detector"
)

print(get_metrics_summary(metrics))
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Example 2: Multi-Class (3-class Radar)

```python
import numpy as np
from src.evaluation_enhanced import compute_comprehensive_metrics

# 3-class radar classification
y_true = np.repeat([0, 1, 2], 30)  # 30 samples per class
np.random.seed(42)

# Simulate predictions with ~85% accuracy
y_pred = y_true.copy()
errors = np.random.choice(90, 15, replace=False)
for idx in errors:
    y_pred[idx] = (y_true[idx] + np.random.randint(1, 3)) % 3

# Generate probabilities
y_probs = np.random.rand(90, 3)
for i in range(90):
    y_probs[i, y_pred[i]] += 1.0
    y_probs[i] /= y_probs[i].sum()

metrics = compute_comprehensive_metrics(
    predictions=y_pred,
    labels=y_true,
    probabilities=y_probs,
    model_name="radar_3class"
)

# Per-class analysis
for i in range(3):
    print(f"Class {i}: Precision={metrics['per_class']['precision'][i]:.4f}, "
          f"Recall={metrics['per_class']['recall'][i]:.4f}, "
          f"F1={metrics['per_class']['f1'][i]:.4f}")
```

### Example 3: PyTorch Model Evaluation

```python
import torch
from src.evaluation_enhanced import evaluate_pytorch_enhanced, get_metrics_summary

# Assume model and test_loader exist
metrics = evaluate_pytorch_enhanced(
    model=model,
    loader=test_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="outputs/reports",
    model_name="radar_model_pytorch"
)

print(get_metrics_summary(metrics))
```

---

## Next Steps

1. **Integrate with reporting module**: Use `metrics['cm']` with `plot_confusion_matrix()`
2. **Track metrics over time**: Save multiple evaluation results for monitoring
3. **Compare model versions**: Use `compare_metrics()` for A/B testing
4. **Export for reports**: JSON format enables integration with external BI tools
5. **Custom analysis**: Use returned dictionary for domain-specific metrics

---

## References

- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ROC-AUC for Multi-class Problems](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [Confusion Matrix Interpretation](https://en.wikipedia.org/wiki/Confusion_matrix)
- [F1 Score and Precision-Recall Trade-off](https://en.wikipedia.org/wiki/Precision_and_recall)

---

## Support

For issues or feature requests:
1. Check examples: `examples_evaluation_enhanced.py`
2. Review docstrings in `src/evaluation_enhanced.py`
3. Examine JSON output structure: `outputs/reports/metrics.json`

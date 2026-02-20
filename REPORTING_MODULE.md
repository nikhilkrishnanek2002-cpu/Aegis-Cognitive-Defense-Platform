# Reporting Module Documentation

## Overview

The `reporting.py` module provides publication-quality visualization functions for the Radar AI project. All plots use scientific styling with IEEE-paper-suitable formatting, automatic file saving, and matplotlib-only dependencies.

**Location:** `src/reporting.py`

## Features

✅ **Publication-Quality Figures**
- 300 DPI rendering for high-quality prints
- Professional scientific styling (seaborn-v0_8-darkgrid)
- IEEE conference paper-suitable titles and labels
- Proper grid, legends, and annotations

✅ **Automatic Directory Creation**
- Save directories created automatically if they don't exist
- Supports PNG, PDF, and EPS formats

✅ **Six Core Functions**
- Confusion matrix visualization (heatmap)
- ROC curve with AUC metric
- Precision-Recall curve with F1 iso-contours
- Training history (loss & accuracy)
- Detection performance vs. SNR
- Tracking RMSE over time

## Installation

No additional dependencies beyond project requirements:
- `matplotlib` (already in requirements.txt)
- `numpy` (already in requirements.txt)
- `scikit-learn` (already in requirements.txt)

## API Reference

### 1. `plot_confusion_matrix(y_true, y_pred, save_path)`

Generates a confusion matrix heatmap for classification evaluation.

**Parameters:**
- `y_true` (array-like): Ground truth labels
- `y_pred` (array-like): Predicted labels
- `save_path` (str): Path to save figure (e.g., "plots/cm.png")

**Example:**
```python
from src.reporting import plot_confusion_matrix
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])

plot_confusion_matrix(y_true, y_pred, "results/confusion_matrix.png")
```

**Output:**
- High-contrast heatmap with cell annotations
- Title: "Confusion Matrix: Classification Performance"
- Color bar showing count values

---

### 2. `plot_roc_curve(y_true, y_prob, save_path)`

Generates Receiver Operating Characteristic curve with AUC metric.

**Parameters:**
- `y_true` (array-like): Ground truth binary labels (0 or 1)
- `y_prob` (array-like): Predicted probabilities for positive class
- `save_path` (str): Path to save figure

**Example:**
```python
from src.reporting import plot_roc_curve

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])
y_prob = np.array([0.9, 0.1, 0.8, 0.85, 0.2, 0.95, 0.3, 0.1, 0.9])

plot_roc_curve(y_true, y_prob, "results/roc_curve.png")
```

**Output:**
- ROC curve (blue) with AUC value in legend
- Random classifier baseline (gray dashed line at AUC=0.5)
- Title: "Receiver Operating Characteristic Curve"

---

### 3. `plot_precision_recall(y_true, y_prob, save_path)`

Generates Precision-Recall curve with F1 score iso-contours.

**Parameters:**
- `y_true` (array-like): Ground truth binary labels (0 or 1)
- `y_prob` (array-like): Predicted probabilities for positive class
- `save_path` (str): Path to save figure

**Example:**
```python
from src.reporting import plot_precision_recall

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])
y_prob = np.array([0.9, 0.1, 0.8, 0.85, 0.2, 0.95, 0.3, 0.1, 0.9])

plot_precision_recall(y_true, y_prob, "results/precision_recall.png")
```

**Output:**
- PR curve (green)
- F1 score iso-contours (light gray lines with labels)
- Title: "Precision-Recall Curve with F1 Score Contours"

---

### 4. `plot_training_history(history, save_path)`

Generates dual-axis plot of training and validation loss/accuracy.

**Parameters:**
- `history` (dict): Dictionary with keys:
  - `'loss'` or `'train_loss'`: Training loss values
  - `'val_loss'`: Validation loss values
  - `'accuracy'` or `'train_acc'`: Training accuracy values (optional)
  - `'val_accuracy'` or `'val_acc'`: Validation accuracy values (optional)
- `save_path` (str): Path to save figure

**Example:**
```python
from src.reporting import plot_training_history

history = {
    'loss': [2.3, 1.8, 1.2, 0.8, 0.5],
    'val_loss': [2.4, 1.9, 1.3, 0.9, 0.6],
    'accuracy': [0.3, 0.5, 0.7, 0.85, 0.92],
    'val_accuracy': [0.28, 0.48, 0.68, 0.82, 0.89],
}

plot_training_history(history, "results/training_history.png")
```

**Output:**
- Left subplot: Training and validation loss (crossentropy)
- Right subplot: Training and validation accuracy
- Title: "Model Training History"

---

### 5. `plot_detection_vs_snr(snr_values, accuracy, save_path)`

Generates detection performance curve as a function of Signal-to-Noise Ratio.

**Parameters:**
- `snr_values` (array-like): SNR values in dB
- `accuracy` (array-like): Detection accuracy at each SNR level (0-1)
- `save_path` (str): Path to save figure

**Example:**
```python
from src.reporting import plot_detection_vs_snr

snr_values = np.array([-5, -3, 0, 3, 5, 8, 10, 12, 15, 18, 20])
accuracy = np.array([0.1, 0.15, 0.25, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97, 0.99, 0.99])

plot_detection_vs_snr(snr_values, accuracy, "results/detection_vs_snr.png")
```

**Output:**
- Detection accuracy vs. SNR plot (blue curve with markers)
- 90% accuracy threshold line (dashed gray)
- Y-axis formatted as percentage
- Title: "Radar Detection Performance vs. SNR"

---

### 6. `plot_tracking_rmse(time, rmse, save_path)`

Generates tracking error (RMSE) visualization over time.

**Parameters:**
- `time` (array-like): Time values (seconds or frame numbers)
- `rmse` (array-like): Root Mean Square Error values (meters or normalized)
- `save_path` (str): Path to save figure

**Example:**
```python
from src.reporting import plot_tracking_rmse

time = np.arange(0, 100, 0.1)
rmse = 5.0 + 2.0 * np.sin(time / 10) + np.random.normal(0, 0.3, len(time))

plot_tracking_rmse(time, rmse, "results/tracking_rmse.png")
```

**Output:**
- RMSE over time (red line)
- Mean RMSE reference line (dashed gray)
- Shaded regions: above mean (red), below mean (green)
- Title: "Target Tracking Root Mean Square Error"

## Integration with Experiment Runner

The reporting module integrates seamlessly with `experiment_runner.py`:

```python
from src.reporting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    plot_detection_vs_snr,
)

# Inside evaluation stage
plot_confusion_matrix(y_test, y_pred_test, f"{output_dir}/confusion_matrix.png")
plot_roc_curve(y_test, y_prob_test, f"{output_dir}/roc_curve.png")
plot_training_history(history, f"{output_dir}/training_history.png")
plot_detection_vs_snr(snr_vals, acc_vals, f"{output_dir}/detection_vs_snr.png")
```

## Output Directory Structure

```
results/
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall.png
│   ├── training_history.png
│   ├── detection_vs_snr.png
│   └── tracking_rmse.png
├── models/
└── logs/
```

## Styling Details

All plots are configured with the following scientific standards:

| Property | Value |
|----------|-------|
| DPI | 300 (publication quality) |
| Style | seaborn-v0_8-darkgrid |
| Figure Size | 8×6 inches (or 14×5 for subplot) |
| Font Family | sans-serif |
| Font Size | 11 pt (body), 12 pt (axes), 13 pt (title) |
| Line Width | 2.5 pt |
| Marker Size | 6 pt |
| Grid | Enabled (alpha=0.3) |
| Format | PNG (300 DPI), PDF, or EPS supported |

## Testing

Run the example script to generate test plots:

```bash
cd /home/nikhil/PycharmProjects/"Aegis Cognitive Defense Platform"
python examples_reporting.py
```

This generates all 6 plot types with synthetic data in `results/reports/`.

## Key IEEE Paper Features

✅ High contrast and readability
✅ Proper mathematical notation in labels (e.g., "Cross-Entropy Loss")
✅ Clear legends with metric values
✅ Minimal chartjunk (clean aesthetics)
✅ Grid lines for ease of reading
✅ 300 DPI for vector-quality prints
✅ Professional color palette
✅ Proper figure spacing and padding

## File Format Support

All functions automatically save in the format specified by `save_path` extension:
- `.png` - Raster format (recommended for presentations)
- `.pdf` - Vector format (recommended for LaTeX papers)
- `.eps` - Encapsulated PostScript (for publishing)

```python
# Save as PNG
plot_confusion_matrix(y_true, y_pred, "confusion.png")

# Save as PDF
plot_roc_curve(y_true, y_prob, "roc_curve.pdf")

# Save as EPS
plot_training_history(history, "training.eps")
```

## Performance Notes

- Plotting is fast: <1 second per figure on CPU
- Memory efficient: Files saved directly to disk, not held in memory
- Compatible with batch processing: Create multiple plots in a loop
- No GPU required: Pure CPU-based visualization

## Common Issues and Solutions

**Issue:** Plots not displaying in Jupyter?
- Use inline backend: `%matplotlib inline`
- The module saves to disk regardless of backend

**Issue:** Directory not found?
- Directories are auto-created, but verify write permissions

**Issue:** Too small on screen?
- Increase figure size in matplotlib (`plt.rcParams['figure.figsize']`)
- Or view PNG files directly (already 300 DPI)

**Issue:** Need different styling?
- Modify `_setup_scientific_style()` function
- Or save to PDF/EPS vector format for post-processing in Inkscape

## Citation

If you use these plots in published research, cite the Radar AI project in your work.

---

**Module Version:** 1.0
**Last Updated:** 2026-02-20
**Compatible with:** Python 3.8+

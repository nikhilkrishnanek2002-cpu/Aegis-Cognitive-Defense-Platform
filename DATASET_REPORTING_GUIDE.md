# Dataset Reporting Utility - Comprehensive Guide

## Overview

The **Dataset Reporting Utility** (`src/dataset_reporting.py`) provides comprehensive analysis and visualization capabilities for radar signal datasets. It generates detailed reports on class distribution, signal characteristics, SNR statistics, and creates publication-quality visualizations.

### Key Features

✅ **Class Distribution Analysis**
- Frequency counting per class
- Percentage breakdown
- Class imbalance detection

✅ **Signal Characteristics**
- Signal length statistics (min, max, mean, median, std)
- Total sample counts per signal
- Per-class signal length analysis

✅ **SNR Distribution (Optional)**
- SNR statistics (min, max, mean, median, Q1, Q3)
- Per-class SNR analysis
- SNR percentile distributions

✅ **Multiple Output Formats**
- **Text Report** — Human-readable summary (`dataset_summary.txt`)
- **CSV Export** — Machine-readable statistics (`dataset_summary.csv`)
- **Visualizations** — Bar charts, histograms, box plots (`dataset_distribution.png`)

✅ **Publication-Quality Plots**
- Class distribution (bar and pie charts)
- Signal length distributions (histograms)
- Per-class signal length variations (box plots)
- SNR distributions and per-class comparison

---

## Installation & Setup

### Dependencies

```bash
pip install numpy matplotlib
```

The module requires:
- `numpy` — Numerical computations
- `matplotlib` — Visualization and plotting

### Quick Start

```python
from src.dataset_reporting import DatasetReporter

# Create reporter with your dataset
reporter = DatasetReporter(
    signals=signals,         # List of signal arrays
    labels=labels,           # Class labels
    class_names=['Class 0', 'Class 1'],  # Optional
    snr_values=snr_array     # Optional
)

# Generate all reports
reporter.generate_all_reports(
    output_dir="outputs/reports",
    plot_dir="outputs/plots"
)

# Output files created:
# - outputs/reports/dataset_summary.txt
# - outputs/reports/dataset_summary.csv
# - outputs/plots/dataset_distribution.png
```

---

## API Reference

### DatasetReporter Class

#### Constructor: `__init__()`

```python
DatasetReporter(
    signals: Union[List[np.ndarray], np.ndarray],
    labels: np.ndarray,
    signal_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    snr_values: Optional[np.ndarray] = None,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `signals` | `Union[List, np.ndarray]` | — | List or array of signal data. Each signal can be 1D or multi-dimensional |
| `labels` | `np.ndarray` | — | Class labels (shape: (n_samples,)) |
| `signal_names` | `Optional[List[str]]` | `None` | Optional names for signals |
| `class_names` | `Optional[List[str]]` | `None` | Optional class names. If `None`, uses 'Class 0', 'Class 1', etc. |
| `snr_values` | `Optional[np.ndarray]` | `None` | SNR values in dB (shape: (n_samples,)). Optional |

#### Example

```python
import numpy as np
from src.dataset_reporting import DatasetReporter

# Create synthetic dataset
signals = [np.random.randn(150), np.random.randn(200), ...]
labels = np.array([0, 1, 0, 1, ...])
snr_values = np.array([10.5, 15.3, 12.1, ...])

# Create reporter
reporter = DatasetReporter(
    signals=signals,
    labels=labels,
    class_names=['Signal', 'Noise'],
    snr_values=snr_values
)
```

---

### Main Methods

#### `compute_statistics() -> Dict[str, Any]`

Compute comprehensive dataset statistics.

**Returns:**
```python
{
    'class_distribution': {0: 50, 1: 50},  # Class -> count
    'signal_lengths': {
        'min': 100,
        'max': 500,
        'mean': 298.5,
        'median': 300.0,
        'std': 85.3,
        'total_samples': 149250,
    },
    'signal_total_samples': {  # For multi-dimensional signals
        'min': 100,
        'max': 500,
        'mean': 298.5,
        'median': 300.0,
        'std': 85.3,
    },
    'snr_stats': {  # If SNR provided
        'min': -10.5,
        'max': 30.2,
        'mean': 12.3,
        'median': 12.5,
        'std': 8.5,
        'q1': 5.2,
        'q3': 19.1,
    },
    'per_class': {
        'Signal': {
            'count': 50,
            'percentage': 50.0,
            'signal_lengths': {'min': 100, 'max': 450, 'mean': 280.0},
            'snr': {'min': -5.0, 'max': 30.2, 'mean': 15.0},
        },
        'Noise': {...},
    }
}
```

**Example:**
```python
stats = reporter.compute_statistics()
print(f"Class distribution: {stats['class_distribution']}")
print(f"Mean signal length: {stats['signal_lengths']['mean']:.1f}")
```

#### `generate_text_report() -> str`

Generate human-readable text report.

**Returns:** Formatted string with all statistics

**Example:**
```python
report = reporter.generate_text_report()
print(report)
```

#### `save_text_report(output_file: str) -> None`

Save text report to file.

**Parameters:**
- `output_file` — Path to output `.txt` file

**Example:**
```python
reporter.save_text_report("outputs/reports/dataset_summary.txt")
```

#### `save_csv_summary(output_file: str) -> None`

Save CSV summary of statistics.

**Parameters:**
- `output_file` — Path to output `.csv` file

**Example:**
```python
reporter.save_csv_summary("outputs/reports/dataset_summary.csv")
```

#### `plot_distributions(output_file: str, figsize: Tuple, dpi: int) -> None`

Generate comprehensive distribution plots.

**Parameters:**
- `output_file` — Path to output PNG file (default: `"outputs/plots/dataset_distribution.png"`)
- `figsize` — Figure size as (width, height) in inches (default: `(16, 10)`)
- `dpi` — Resolution in dots per inch (default: `300`)

**Example:**
```python
reporter.plot_distributions(
    output_file="outputs/plots/dataset_distribution.png",
    figsize=(16, 10),
    dpi=300
)
```

**Generates 6 subplots:**
1. **Class Distribution (Bar)** — Count per class with labels
2. **Class Distribution (Pie)** — Percentage breakdown
3. **Signal Length Histogram** — Distribution with mean/median lines
4. **Signal Length Box Plot** — Per-class variation
5. **SNR Histogram** — Distribution with statistics (if available)
6. **SNR Box Plot** — Per-class SNR comparison (if available)

#### `generate_all_reports(output_dir: str, plot_dir: Optional[str]) -> Dict[str, str]`

Generate all reports (text, CSV, plots) in one call.

**Parameters:**
- `output_dir` — Directory for text and CSV files (default: `"outputs/reports"`)
- `plot_dir` — Directory for plot files (default: `"outputs/plots"`)

**Returns:** Dictionary with paths to generated files

**Example:**
```python
paths = reporter.generate_all_reports(
    output_dir="outputs/reports",
    plot_dir="outputs/plots"
)

print(paths['text_report'])   # outputs/reports/dataset_summary.txt
print(paths['csv_summary'])   # outputs/reports/dataset_summary.csv
print(paths['plots'])         # outputs/plots/dataset_distribution.png
```

---

## Output File Formats

### Text Report (`dataset_summary.txt`)

Example output:
```
================================================================================
                             DATASET SUMMARY REPORT
================================================================================

Generated: 2026-02-20T10:14:44.788788
Total Samples: 198
Number of Classes: 3
SNR Data Available: Yes

────────────────────────────────────────────────────────────────────────────────
CLASS DISTRIBUTION
────────────────────────────────────────────────────────────────────────────────
Target A                 66 samples ( 33.3%)
Target B                 66 samples ( 33.3%)
Clutter                  66 samples ( 33.3%)

────────────────────────────────────────────────────────────────────────────────
SIGNAL LENGTH STATISTICS (Primary Dimension)
────────────────────────────────────────────────────────────────────────────────
Minimum:       102 samples
Maximum:       495 samples
Mean:          308.0 samples
Median:        313.5 samples
Std Dev:       114.8 samples
Total:         60,989 samples
...
```

### CSV Summary (`dataset_summary.csv`)

Example output:
```
Dataset Statistics Summary
Generated,2026-02-20T10:14:44.788788
Total Samples,198
Number of Classes,3
SNR Available,Yes

Class Distribution
Class,Count,Percentage
Target A,66,33.33%
Target B,66,33.33%
Clutter,66,33.33%

Signal Length Statistics
Metric,Value
min,102
max,495
mean,308.02525252525254
median,313.5
std,114.77089329690064
total_samples,60989
...
```

### Visualization (`dataset_distribution.png`)

6-panel figure with:
1. Class distribution bar chart with value labels
2. Class distribution pie chart with percentages
3. Signal length histogram with mean/median overlays
4. Per-class signal length box plots
5. SNR distribution histogram (if available)
6. Per-class SNR box plots (if available)

---

## Use Cases

### Use Case 1: Dataset Validation Before Training

```python
from src.dataset_reporting import DatasetReporter

# Load your training dataset
signals, labels, snr_values = load_training_data()

# Generate reports
reporter = DatasetReporter(signals, labels, snr_values=snr_values)
reporter.generate_all_reports()

# Check for issues
stats = reporter.compute_statistics()
for class_name, stat in stats['per_class'].items():
    if stat['count'] < 10:
        print(f"⚠️ {class_name} has only {stat['count']} samples!")
```

### Use Case 2: Dataset Comparison

```python
# Compare training and test sets
train_reporter = DatasetReporter(train_signals, train_labels)
test_reporter = DatasetReporter(test_signals, test_labels)

train_stats = train_reporter.compute_statistics()
test_stats = test_reporter.compute_statistics()

# Compare distributions
for class_name in train_stats['per_class']:
    train_pct = train_stats['per_class'][class_name]['percentage']
    test_pct = test_stats['per_class'][class_name]['percentage']
    print(f"{class_name}: Train {train_pct:.1f}% vs Test {test_pct:.1f}%")
```

### Use Case 3: Dataset Characteristics Report

```python
# Generate report for documentation
reporter = DatasetReporter(signals, labels, snr_values=snr_values)

report_text = reporter.generate_text_report()
with open('dataset_report.txt', 'w') as f:
    f.write(report_text)

reporter.save_csv_summary('dataset_stats.csv')
reporter.plot_distributions('dataset_plots.png')

# All files ready for publication/documentation
```

### Use Case 4: Analysis Without Plotting

```python
# For large datasets, skip plots and get statistics only
reporter = DatasetReporter(signals, labels, snr_values=snr_values)

# Get statistics directly
stats = reporter.compute_statistics()

# Save text report (fast)
reporter.save_text_report('dataset_summary.txt')

# Save CSV (fast)
reporter.save_csv_summary('dataset_summary.csv')

# Skip plotting to save time
```

---

## Integration Patterns

### Pattern 1: Post-Training Dataset Analysis

```python
from src.dataset_reporting import DatasetReporter
from src.train_pytorch import train_pytorch_model

# Train model
model, history = train_pytorch_model(epochs=20, output_dir="results")

# Analyze the training dataset
reporter = DatasetReporter(
    signals=train_signals,
    labels=train_labels,
    snr_values=train_snr,
    class_names=['Class A', 'Class B', 'Class C']
)

reporter.generate_all_reports()

# Saves to:
# - outputs/reports/dataset_summary.txt
# - outputs/reports/dataset_summary.csv  
# - outputs/plots/dataset_distribution.png
```

### Pattern 2: Pipeline Integration

```python
def prepare_and_analyze_dataset(data_config):
    """Load, prepare, and analyze dataset."""
    from src.dataset_reporting import DatasetReporter
    
    # Load data
    signals = load_signals(data_config['path'])
    labels = load_labels(data_config['path'])
    snr_values = compute_snr(signals)
    
    # Create reporter
    reporter = DatasetReporter(
        signals,
        labels,
        class_names=data_config['class_names'],
        snr_values=snr_values
    )
    
    # Generate reports
    reporter.generate_all_reports()
    
    # Check for issues
    stats = reporter.compute_statistics()
    
    # Validate class distribution
    for class_name, stat in stats['per_class'].items():
        if stat['percentage'] < 5:
            raise ValueError(f"Class {class_name} is under-represented")
    
    return reporter, stats
```

### Pattern 3: Batch Dataset Analysis

```python
from src.dataset_reporting import DatasetReporter

datasets = {
    'training': (train_signals, train_labels),
    'validation': (val_signals, val_labels),
    'test': (test_signals, test_labels),
}

for split_name, (signals, labels) in datasets.items():
    reporter = DatasetReporter(signals, labels)
    
    reporter.generate_all_reports(
        output_dir=f"outputs/reports/{split_name}",
        plot_dir=f"outputs/plots/{split_name}"
    )
    
    print(f"✓ {split_name} dataset analyzed")
```

---

## Statistics Explained

### Class Distribution
- **Count**: Number of samples per class
- **Percentage**: Proportion of total dataset

**Use:** Detect class imbalance

### Signal Length Statistics
- **Min/Max**: Range of signal lengths
- **Mean**: Average signal length
- **Median**: Middle value (robust to outliers)
- **Std Dev**: Variability of lengths
- **Total**: Sum of all samples across signals

**Use:** Understand signal dimensionality, detect padding needs

### SNR Statistics (Signal-to-Noise Ratio)
- **Min/Max**: SNR range in dB
- **Mean/Median**: Central tendency
- **Q1/Q3**: Quartiles (25th/75th percentile)
- **Std Dev**: Variability

**Use:** Assess dataset difficulty, evaluate robustness testing

### Per-Class Statistics
Per-class metrics for:
- Sample counts and percentages
- Signal length ranges
- SNR characteristics

**Use:** Identify class-specific characteristics and imbalances

---

## Performance Considerations

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Statistics computation | ~0.1s | Low | Independent of visualization |
| Text report generation | ~0.01s | Low | Very fast |
| CSV export | ~0.02s | Low | Negligible overhead |
| Plot generation | ~2-5s | Medium | Scales with signal count |
| All reports | ~3-6s | Medium | Plotting dominates time |

**Recommendations:**
- For quick analysis: Use `compute_statistics()` only
- For large datasets (>10k signals): Skip plots, use statistics
- For final reports: Generate all reports with high DPI (300)
- For quick checks: Use lower DPI (100-150) for plots

---

## Troubleshooting

### Issue: "Signal and labels must have same length"
**Solution:** Ensure signals and labels are aligned:
```python
assert len(signals) == len(labels)
```

### Issue: "Class names don't match number of classes"
**Solution:** Provide class names for all classes found in data:
```python
n_classes = len(np.unique(labels))
reporter = DatasetReporter(signals, labels, class_names=['A', 'B', 'C'])
```

### Issue: Output directory doesn't exist
**Solution:** Module auto-creates directories, but ensure you have write permissions:
```python
import os
os.makedirs("outputs/reports", exist_ok=True)
```

### Issue: SNR plots showing "Not Available"
**Solution:** To include SNR plots, provide SNR data:
```python
reporter = DatasetReporter(signals, labels, snr_values=snr_array)
```

### Issue: Plots are too small/unreadable
**Solution:** Increase figure size and DPI:
```python
reporter.plot_distributions(figsize=(20, 12), dpi=300)
```

---

## Examples

### Example 1: Basic Analysis

```python
import numpy as np
from src.dataset_reporting import DatasetReporter

# Create dataset
signals = [np.random.randn(100) for _ in range(100)]
labels = np.repeat([0, 1], 50)

# Analyze
reporter = DatasetReporter(signals, labels, class_names=['A', 'B'])
reporter.generate_all_reports()
```

### Example 2: With SNR Data

```python
import numpy as np
from src.dataset_reporting import DatasetReporter

signals = [np.random.randn(200) for _ in range(150)]
labels = np.tile([0, 1, 2], 50)
snr_values = np.random.uniform(-5, 30, 150)

reporter = DatasetReporter(
    signals=signals,
    labels=labels,
    class_names=['Radar A', 'Radar B', 'Clutter'],
    snr_values=snr_values
)

reporter.generate_all_reports(
    output_dir="outputs/reports",
    plot_dir="outputs/plots"
)
```

### Example 3: Statistics Only

```python
from src.dataset_reporting import DatasetReporter

reporter = DatasetReporter(signals, labels, snr_values=snr_values)
stats = reporter.compute_statistics()

# Print class distribution
for class_name, stat in stats['per_class'].items():
    print(f"{class_name}: {stat['count']} samples ({stat['percentage']:.1f}%)")

# Check SNR
if 'snr_stats' in stats:
    snr = stats['snr_stats']
    print(f"SNR range: {snr['min']:.1f} - {snr['max']:.1f} dB")
```

---

## Next Steps

1. **Analyze your dataset** — Run `DatasetReporter` on your data
2. **Check for issues** — Review class distribution and signal characteristics
3. **Generate reports** — Create documentation-quality outputs
4. **Iterate** — Fix dataset issues and re-analyze
5. **Compare datasets** — Use statistics to compare train/val/test splits

---

## References

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Visualization](https://matplotlib.org/)
- [Statistical Measures](https://en.wikipedia.org/wiki/Descriptive_statistics)
- [Class Imbalance](https://en.wikipedia.org/wiki/Class_imbalance)

---

## Support

For issues or feature requests:
1. Check examples: `examples_dataset_reporting.py`
2. Review docstrings in `src/dataset_reporting.py`
3. Examine output files: `outputs/reports/` and `outputs/plots/`

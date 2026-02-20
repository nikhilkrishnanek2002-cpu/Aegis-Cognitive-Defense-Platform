# Experiment Runner Documentation

## Overview

The `experiment_runner.py` is a comprehensive orchestrator for the Cognitive Radar AI project that automates the end-to-end machine learning pipeline:

1. ✅ **Data Preprocessing** - Generates synthetic radar datasets
2. ✅ **Model Training** - Trains the PhotonicRadarAI model using PyTorch
3. ✅ **Model Evaluation** - Computes accuracy, detection metrics, and creates visualizations
4. ✅ **Results Persistence** - Saves models, metrics, and training histories

## Features

- **Seeded Randomness**: Sets global seeds for numpy, torch, and Python for reproducibility
- **Organized Output**: Creates timestamped experiment directories with organized subdirectories:
  ```
  outputs/exp_YYYYMMDD_HHMMSS/
  ├── models/          (trained model files)
  ├── logs/            (experiment logs)
  ├── plots/           (visualizations)
  └── reports/         (metrics, config, history)
  ```
- **Dual Logging**: Logs both to console (INFO) and files
- **GPU Support**: Automatically detects and uses CUDA if available
- **YAML Configuration**: Flexible configuration via YAML files
- **Comprehensive Metrics**: Tracks accuracy, probability of detection (Pd), and false alarm rate (FAR)

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- `torch` - Deep learning framework
- `numpy` - Numerical computing
- `pyyaml` - Configuration files
- `scikit-learn` - Metrics and utilities
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `opencv-python` - Image operations

## Usage

### Basic Usage

Run with default configuration (`config.yaml`):

```bash
python experiment_runner.py
```

### Custom Configuration

Run with a specific configuration file:

```bash
python experiment_runner.py --config experiment_config_example.yaml
```

Or with a custom path:

```bash
python experiment_runner.py --config /path/to/custom_config.yaml
```

## Configuration

Create a YAML configuration file with the following structure:

```yaml
experiment:
  name: "my_experiment"
  description: "Description of your experiment"
  seed: 42                    # Random seed for reproducibility
  output_dir: outputs         # Base output directory
  samples_per_class: 50       # Samples per class for dataset generation

training:
  epochs: 20                  # Number of training epochs
  batch_size: 16              # Batch size
  learning_rate: 0.001        # Learning rate for Adam optimizer
  validation_split: 0.2       # Validation split ratio

model_config:
  num_classes: 6              # Number of output classes
  metadata_size: 8            # Metadata feature dimension
  input_height: 128           # Input image height
  input_width: 128            # Input image width

logging:
  level: INFO                 # Log level: DEBUG, INFO, WARNING, ERROR
  dir: results                # Log directory
  file: system.log            # Log filename
  max_bytes: 10485760         # Max log file size (10MB)
  backup_count: 5             # Number of backup log files
```

### Example Configuration Files

- **`experiment_config_example.yaml`** - Minimal example configuration
- **`config.yaml`** - Full system configuration with all subsystems

## Output Structure

Each experiment run creates a timestamped directory:

```
outputs/exp_20260220_143022/
│
├── models/
│   └── model_final.pt              # Trained model weights
│
├── logs/
│   └── experiment.log              # Detailed experiment log
│
├── plots/
│   └── confusion_matrix.png        # Classification confusion matrix
│
└── reports/
    ├── metrics.json                # Final metrics (accuracy, Pd, FAR)
    ├── training_history.json       # Per-epoch training losses
    ├── config.yaml                 # Copy of configuration used
    └── (additional reports)
```

## Output Files

### `metrics.json`
Contains final evaluation metrics:
```json
{
  "accuracy": 0.9167,
  "probability_of_detection": 0.9167,
  "false_alarm_rate": 0.0833,
  "confusion_matrix": [[...], [...], ...]
}
```

### `training_history.json`
Contains per-epoch training data:
```json
{
  "epoch": [1, 2, 3, ...],
  "loss": [2.1543, 1.8932, 1.6234, ...]
}
```

### `experiment.log`
Detailed log of all pipeline stages with timestamps and status information.

## Pipeline Stages

### Stage 1: Data Preprocessing
- Generates synthetic radar signals for 6 classes:
  - drone
  - aircraft
  - bird
  - helicopter
  - missile
  - clutter
- Extracts features:
  - Range-Doppler maps
  - Spectrograms
  - Photonic metadata
- Resizes to 128×128 and normalizes

### Stage 2: Model Training
- Loads preprocessing data into PyTorch DataLoader
- Initializes PhotonicRadarAI model with:
  - 2 CNN branches (Range-Doppler + Spectrogram)
  - 1 MLP branch (metadata)
  - Feature fusion layer
  - Classification head
- Trains with Adam optimizer and CrossEntropyLoss
- Reports loss every N epochs

### Stage 3: Model Evaluation
- Evaluates model on full dataset
- Computes:
  - **Accuracy** - Overall classification accuracy
  - **Probability of Detection (Pd)** - Trace/sum of confusion matrix
  - **False Alarm Rate (FAR)** - Off-diagonal sum/total
- Generates confusion matrix heatmap visualization
- Logs all metrics

### Stage 4: Results Saving
- Saves trained model state dict to `.pt` file
- Saves metrics to JSON
- Saves training history to JSON
- Saves configuration copy for reproducibility

## GPU Support

The runner automatically detects GPU availability:

```
[DEVICE] Using GPU: NVIDIA GeForce RTX 3090
```

If CUDA is unavailable, it falls back to CPU:

```
[DEVICE] Using CPU
```

## Logging

All events are logged to both console and file:

```
2026-02-20 14:30:22 - experiment - INFO - Experiment started at 2026-02-20 14:30:22.123456
2026-02-20 14:30:22 - experiment - INFO - Config loaded from: config.yaml
2026-02-20 14:30:22 - experiment - INFO - ================================================== ====
2026-02-20 14:30:22 - experiment - INFO - STAGE 1: DATA PREPROCESSING
...
```

## Error Handling

If an error occurs, the runner will:
1. Log the full exception traceback to the log file
2. Print the error to stderr
3. Exit with status code 1

Example error output:
```
ERROR: Experiment failed - [specific error message]
```

Check `outputs/exp_*/logs/experiment.log` for detailed error information.

## Performance Metrics

The runner reports these key metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct classifications |
| **Probability of Detection (Pd)** | True positive rate (trace of confusion matrix) |
| **False Alarm Rate (FAR)** | False positive rate |

Ideal performance: High Pd, Low FAR

## Advanced Usage

### Hyperparameter Sweeps

Create multiple config files for different hyperparameters:

```bash
for lr in 0.0001 0.001 0.01; do
  cat > config_lr${lr}.yaml << EOF
training:
  learning_rate: $lr
  epochs: 20
EOF
  python experiment_runner.py --config config_lr${lr}.yaml
done
```

### Integration with External Workflows

The runner can be integrated into CI/CD pipelines:

```bash
python experiment_runner.py --config config.yaml || exit 1
# Post-process results
python analyze_results.py outputs/exp_*
```

## Troubleshooting

### Config file not found
```
FileNotFoundError: Config file not found: path/to/config.yaml
```
**Solution**: Verify the config file path and ensure it exists.

### Out of memory error
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce `batch_size` in config
- Reduce `samples_per_class` in config
- Use CPU instead (no GPU required)

### Import errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Slow training
If training is slow on GPU, ensure PyTorch is compiled with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

## Example Workflow

1. **Create experiment config**:
   ```bash
   cp experiment_config_example.yaml my_experiment.yaml
   # Edit my_experiment.yaml with custom parameters
   ```

2. **Run experiment**:
   ```bash
   python experiment_runner.py --config my_experiment.yaml
   ```

3. **Monitor progress**:
   ```bash
   # In another terminal:
   tail -f outputs/exp_*/logs/experiment.log
   ```

4. **Analyze results**:
   ```bash
   # View metrics
   cat outputs/exp_*/reports/metrics.json | python -m json.tool
   
   # Visualize training loss
   python -c "
   import json
   with open('outputs/exp_*/reports/training_history.json') as f:
       history = json.load(f)
       import matplotlib.pyplot as plt
       plt.plot(history['epoch'], history['loss'])
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.savefig('training_loss.png')
   "
   ```

5. **Load trained model for inference**:
   ```python
   import torch
   from src.model_pytorch import build_pytorch_model
   
   model = build_pytorch_model()
   model.load_state_dict(torch.load('outputs/exp_*/models/model_final.pt'))
   model.eval()
   # Use model for inference
   ```

## Key Classes and Functions

### `ExperimentRunner`
Main orchestrator class with methods:
- `_load_config()` - Load YAML config
- `_set_seeds()` - Set reproducibility seeds  
- `_create_output_dirs()` - Create directory structure
- `_setup_logging()` - Initialize logging
- `_setup_device()` - Setup CUDA/CPU
- `_preprocess_data()` - Run preprocessing
- `_train_model()` - Run training loop
- `_evaluate_model()` - Run evaluation
- `_save_results()` - Persist results
- `_print_summary()` - Print experiment summary
- `run()` - Execute full pipeline

### Integration Module Functions
- `create_pytorch_dataset()` - Generate synthetic radar data
- `build_pytorch_model()` - Initialize PhotonicRadarAI model
- `init_logging()` - Setup logger (if used)


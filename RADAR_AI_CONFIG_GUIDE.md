# Radar AI Experiment Configuration Guide

## Overview

The `radar_ai_experiment.yaml` file is a production-grade configuration for reproducible research experiments with the Cognitive Photonic Radar AI system. It includes 9 major sections with 100+ configurable parameters.

---

## File Structure

```
radar_ai_experiment.yaml
├── experiment          (Metadata & experiment info)
├── dataset             (Data paths, generation, preprocessing)
├── signal_simulation   (Radar signal parameters, photonic effects)
├── model               (Architecture: CNN branches, MLP, fusion)
├── training            (Epochs, batch size, optimizer, LR schedule)
├── evaluation          (Metrics, visualizations, radar-specific metrics)
├── output              (Directory structure, file formats)
├── logging             (Console/file logging, experiment tracking)
├── reproducibility     (Seeds, determinism, validation)
├── performance         (Device, profiling, precision)
└── notes               (Documentation and expected results)
```

---

## Quick Start

### Use Default Configuration
```bash
# Runs with all default parameters
python experiment_runner.py --config radar_ai_experiment.yaml
```

### Customize for Your Experiment
```bash
# Copy and modify
cp radar_ai_experiment.yaml my_experiment.yaml

# Edit specific sections, then run
python experiment_runner.py --config my_experiment.yaml
```

---

## Key Sections Explained

### 1. EXPERIMENT METADATA
```yaml
experiment:
  name: "cognitive_radar_baseline"
  description: "Baseline experiment..."
  seed: 42                    # Master seed for reproducibility
  output_dir: experiments     # Base output directory
  tags: ["radar", "photonic"] # For organization
```

**Why it matters**: Enables tracking and reproducibility. The seed ensures identical results across runs.

---

### 2. DATASET CONFIGURATION
```yaml
dataset:
  samples_per_class: 50       # Synthetic data generation
  n_classes: 6                # 6 radar target types
  train_split: 0.7            # 70/15/15 train/val/test split
  preprocessing:
    resize_height: 128
    resize_width: 128
    normalize: true
    augmentation: false
```

**Key parameters**:
- `samples_per_class`: More samples = better models but slower training
- `train_split`: Adjust based on your data availability
- `augmentation`: Enable for smaller datasets

**Example variations**:
```yaml
# For small dataset (data-constrained)
samples_per_class: 25
augmentation: true

# For large dataset (compute-constrained)
samples_per_class: 200
augmentation: false
```

---

### 3. SIGNAL SIMULATION
```yaml
signal_simulation:
  carrier_frequency_hz: 10e9      # 10 GHz
  bandwidth_hz: 5e9              # 5 GHz bandwidth
  target_snr_db: 10.0            # Signal-to-noise ratio
  photonic_enabled: true         # Enable photonic model
  laser_linewidth_hz: 1e4        # Phase noise
  temperature_drift_enabled: true # Thermal effects
```

**Critical parameters**:
- `target_snr_db`: Lower SNR = harder problem (more robust model)
  - Easy: 20 dB
  - Medium: 10 dB
  - Hard: 0 dB
  - Extreme: -5 dB

- `photonic_enabled`: Include realistic photonic effects?
  - `true`: More realistic, harder problem
  - `false`: Simpler, faster training

- `laser_linewidth_hz`: Phase noise source
  - `1e4` (10 kHz) = moderate noise
  - `1e5` (100 kHz) = high noise (challenging)
  - `1e3` (1 kHz) = low noise (easy)

**Example variations**:
```yaml
# Easy baseline (good starting point)
target_snr_db: 15.0
photonic_enabled: false
laser_linewidth_hz: 1e3

# Medium difficulty (realistic)
target_snr_db: 10.0
photonic_enabled: true
laser_linewidth_hz: 1e4

# Hard scenario (robustness testing)
target_snr_db: 0.0
photonic_enabled: true
laser_linewidth_hz: 1e5
```

---

### 4. MODEL ARCHITECTURE
```yaml
model:
  rd_branch:          # CNN for Range-Doppler map
    conv_layers:
      - {out_channels: 32, kernel_size: 3, padding: 1}
      - {out_channels: 64, kernel_size: 3, padding: 1}
    dropout: 0.3
  
  spec_branch:        # CNN for Spectrogram
    conv_layers: [...]
  
  metadata_branch:    # MLP for metadata (8-dim vector)
    hidden_layers:
      - {out_dim: 32, activation: "relu"}
      - {out_dim: 16, activation: "relu"}
  
  fusion:             # Combine all branches
    method: "concatenate"
    hidden_dim: 128
```

**Customization examples**:

```yaml
# Smaller model (for faster training)
rd_branch:
  conv_layers:
    - {out_channels: 16, kernel_size: 3, padding: 1}
    - {out_channels: 32, kernel_size: 3, padding: 1}
  dropout: 0.2

# Larger model (for better accuracy)
rd_branch:
  conv_layers:
    - {out_channels: 32, kernel_size: 3, padding: 1}
    - {out_channels: 64, kernel_size: 3, padding: 1}
    - {out_channels: 128, kernel_size: 3, padding: 1}
  dropout: 0.4
```

---

### 5. TRAINING CONFIGURATION
```yaml
training:
  epochs: 20                    # Number of training epochs
  batch_size: 16                # Training batch size
  optimizer:
    name: "adam"
    learning_rate: 0.001
    weight_decay: 1e-5
  
  early_stopping:
    enabled: false              # Stop if no improvement
    patience: 5                 # After 5 epochs with no improvement
  
  lr_schedule:
    enabled: false              # Learning rate decay
    strategy: "step"
    step_size: 10               # Decay every 10 epochs
```

**Customization examples**:

```yaml
# Fast training (quick baseline)
epochs: 10
batch_size: 32
learning_rate: 0.01

# Careful training (best accuracy)
epochs: 100
batch_size: 8
learning_rate: 0.0001
early_stopping:
  enabled: true
  patience: 10

# With learning rate decay
lr_schedule:
  enabled: true
  strategy: "step"
  step_size: 5
  gamma: 0.5
```

---

### 6. EVALUATION CONFIGURATION
```yaml
evaluation:
  eval_every_n_epochs: 1       # Validate after each epoch
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "confusion_matrix"
    - "roc_auc"
  
  radar_metrics:
    probability_of_detection: true   # Pd
    false_alarm_rate: true           # FAR
  
  visualizations:
    confusion_matrix: true
    roc_curve: true
    precision_recall_curve: false
```

**Key radar metrics**:
- **Pd (Probability of Detection)** = Trace(CM) / Sum(CM)
  - How many targets are correctly detected
  - Want: High (>90%)

- **FAR (False Alarm Rate)** = (Sum(CM) - Trace(CM)) / Sum(CM)
  - How many false alarms occur
  - Want: Low (<10%)

---

### 7. OUTPUT CONFIGURATION
```yaml
output:
  base_dir: "experiments"
  create_timestamp_subdir: true      # exp_20260220_143022/
  subdirectories:
    - "models"                       # Saved model weights
    - "logs"                         # Training logs
    - "plots"                        # Visualizations
    - "reports"                      # Metrics, history
  
  save_model: true
  save_metrics: true
  save_history: true
  save_plots: true
```

**Output files generated**:
```
experiments/exp_20260220_143022/
├── models/model_final.pt           # Trained model
├── logs/experiment.log             # Training log
├── plots/confusion_matrix.png      # Visualization
└── reports/
    ├── metrics.json                # Performance metrics
    ├── training_history.json       # Per-epoch loss
    ├── config.yaml                 # Config copy
    └── predictions.json            # Test predictions
```

---

### 8. LOGGING CONFIGURATION
```yaml
logging:
  console:
    enabled: true               # Print to console
    level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  
  file:
    enabled: true              # Write to file
    filename: "experiment.log"
    max_bytes: 10485760        # 10 MB
  
  experiment_tracking:
    enabled: false             # Enable MLflow/W&B (optional)
    backend: "mlflow"
```

---

### 9. REPRODUCIBILITY
```yaml
reproducibility:
  random_seed: 42              # Master seed
  numpy_seed: 42
  torch_seed: 42
  cuda_seed: 42
  
  set_deterministic: true      # Deterministic algorithms
  benchmark: false             # cuDNN determinism
  
  validate_config: true
  validate_data: true
  validate_model: true
```

**Why it matters**: With these settings, running the same config twice produces identical results.

---

## Common Experiment Configurations

### Configuration 1: Quick Baseline (5 minutes)
```yaml
experiment:
  name: "quick_baseline"
  seed: 42

dataset:
  samples_per_class: 25
  
training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.01

signal_simulation:
  photonic_enabled: false
  target_snr_db: 15.0
```

### Configuration 2: Realistic Scenario (30 minutes)
```yaml
experiment:
  name: "realistic_scenario"
  seed: 42

dataset:
  samples_per_class: 50
  
training:
  epochs: 20
  batch_size: 16
  learning_rate: 0.001
  early_stopping:
    enabled: true
    patience: 5

signal_simulation:
  photonic_enabled: true
  target_snr_db: 10.0
  laser_linewidth_hz: 1e4
```

### Configuration 3: Challenging (1 hour)
```yaml
experiment:
  name: "challenging_scenario"
  seed: 42

dataset:
  samples_per_class: 100
  augmentation: true
  
training:
  epochs: 50
  batch_size: 8
  learning_rate: 0.0001
  early_stopping:
    enabled: true
    patience: 10
  lr_schedule:
    enabled: true
    strategy: "step"
    step_size: 10

signal_simulation:
  photonic_enabled: true
  target_snr_db: 0.0
  laser_linewidth_hz: 1e5
  temperature_drift_enabled: true
```

### Configuration 4: Hyperparameter Search
```yaml
# Create multiple configs with different hyperparameters:
# config_lr0001.yaml, config_lr001.yaml, config_lr01.yaml

# Then run sweep:
for config in config_lr*.yaml; do
  python experiment_runner.py --config $config
done
```

---

## Parameter Sensitivity Guide

| Parameter | Effect | Low | Medium | High |
|-----------|--------|-----|--------|------|
| **samples_per_class** | Dataset size | 10 | 50 | 200 |
| **learning_rate** | Optimization speed | 1e-5 | 1e-3 | 1e-1 |
| **batch_size** | Memory & convergence | 4 | 16 | 64 |
| **target_snr_db** | Problem difficulty | 20 | 10 | -5 |
| **epochs** | Training time | 5 | 20 | 100 |
| **dropout** | Regularization | 0.1 | 0.3 | 0.5 |
| **laser_linewidth_hz** | Noise level | 1e3 | 1e4 | 1e5 |

---

## Tips for Research Experiments

### 1. Start Simple
```yaml
# Begin with easy baseline
epochs: 10
batch_size: 32
learning_rate: 0.01
target_snr_db: 15.0
photonic_enabled: false
```

### 2. Increase Complexity Gradually
```yaml
# Then make it realistic
epochs: 20
batch_size: 16
learning_rate: 0.001
target_snr_db: 10.0
photonic_enabled: true
```

### 3. Test Robustness
```yaml
# Finally, test hard scenarios
epochs: 50
batch_size: 8
learning_rate: 0.0001
target_snr_db: 0.0
photonic_enabled: true
temperature_drift_enabled: true
```

### 4. Compare Results
```bash
# Run multiple configurations
python experiment_runner.py --config baseline.yaml
python experiment_runner.py --config realistic.yaml
python experiment_runner.py --config hard.yaml

# Compare metrics
ls -t experiments/exp_*/reports/metrics.json | head -3 | xargs cat
```

---

## Validation Checklist

- [ ] All file paths exist (data/train, data/val, data/test)
- [ ] Seed is set for reproducibility
- [ ] Output directory is writable
- [ ] Batch size fits in GPU memory
- [ ] Number of epochs is reasonable for your hardware
- [ ] Evaluation metrics make sense for your problem
- [ ] Logging is configured appropriately

---

## Example Usage

### Run with custom config
```bash
python experiment_runner.py --config radar_ai_experiment.yaml
```

### Monitor in real-time
```bash
tail -f experiments/exp_*/logs/experiment.log
```

### View results
```bash
cat experiments/exp_*/reports/metrics.json | python -m json.tool
```

### Compare multiple runs
```bash
for dir in experiments/exp_*/; do
  echo "=== $(basename $dir) ==="
  cat "$dir/reports/metrics.json" | grep -E "accuracy|probability_of_detection|false_alarm_rate"
done
```

---

## Advanced Customization

### Multi-GPU Training
```yaml
performance:
  device: "cuda"
  distributed_training: true
  distributed_backend: "nccl"
  distributed_world_size: 4  # 4 GPUs
```

### Experiment Tracking (MLflow)
```yaml
logging:
  experiment_tracking:
    enabled: true
    backend: "mlflow"
    project: "cognitive-radar-ai"
    log_hyperparameters: true
    log_metrics: true
```

### Mixed Precision (FP16)
```yaml
training:
  mixed_precision: true

performance:
  dtype: "float16"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too slow | Reduce `samples_per_class`, increase `batch_size`, fewer epochs |
| Out of memory | Reduce `batch_size`, reduce model size |
| Poor accuracy | Increase `samples_per_class`, enable `photonic_enabled: false` first |
| Non-reproducible | Check all seeds are set, set `deterministic: true` |
| Config errors | Validate YAML syntax, check indentation |

---

## File Reference

- **Main config**: `radar_ai_experiment.yaml`
- **Example config**: `experiment_config_example.yaml`
- **System config**: `config.yaml`
- **Runner**: `experiment_runner.py`


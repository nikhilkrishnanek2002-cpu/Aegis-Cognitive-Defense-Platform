# Radar AI Experiment Configuration - Quick Reference Card

## 📋 Essential Parameters at a Glance

### Data Configuration
```yaml
dataset:
  samples_per_class: [10, 25, 50, 100, 200]      # Data volume
  n_classes: 6                                     # 6 target types (fixed)
  train_split: 0.7                                # 70% train
  preprocessing:
    resize: 128x128                              # Feature array size
    normalize: true                              # Standard practice
    augmentation: [true/false]                   # For small datasets
```

### Signal Difficulty Levels

```
Easy        | Realistic     | Challenging
────────────┼───────────────┼──────────────
SNR: 15 dB  | SNR: 10 dB    | SNR: 0 dB
No photonic | Photonic ON   | Photonic ON
            |               | + Thermal drift
```

### Model Architecture Presets

#### Small (Fast Training)
```yaml
model:
  rd_branch:
    conv_layers:
      - {out_channels: 16, kernel_size: 3}
      - {out_channels: 32, kernel_size: 3}
    dropout: 0.2
  fusion: {hidden_dim: 64}
```

#### Medium (Balanced)
```yaml
model:
  rd_branch:
    conv_layers:
      - {out_channels: 32, kernel_size: 3}
      - {out_channels: 64, kernel_size: 3}
    dropout: 0.3
  fusion: {hidden_dim: 128}
```

#### Large (Best Accuracy)
```yaml
model:
  rd_branch:
    conv_layers:
      - {out_channels: 32, kernel_size: 3}
      - {out_channels: 64, kernel_size: 3}
      - {out_channels: 128, kernel_size: 3}
    dropout: 0.4
  fusion: {hidden_dim: 256}
```

### Training Hyperparameters

| Scenario | Epochs | Batch Size | Learning Rate | Time |
|----------|--------|-----------|---------------|------|
| **Quick Test** | 5 | 32 | 0.01 | ~5 min |
| **Standard** | 20 | 16 | 0.001 | ~30 min |
| **Careful** | 50 | 8 | 0.0001 | ~2 hrs |
| **Research** | 100 | 4 | 0.00001 | ~8 hrs |

### Signal Simulation Parameters

| Parameter | Easy | Medium | Hard |
|-----------|------|--------|------|
| **target_snr_db** | 15 | 10 | 0 |
| **photonic_enabled** | false | true | true |
| **laser_linewidth_hz** | 1e3 | 1e4 | 1e5 |
| **temp_drift_enabled** | false | false | true |
| **clutter_level_db** | -30 | -20 | -10 |

### Optimizer Options

```yaml
# Fast convergence (SGD with momentum)
optimizer:
  name: "sgd"
  learning_rate: 0.01
  momentum: 0.9
  nesterov: true

# Standard (Adam)
optimizer:
  name: "adam"
  learning_rate: 0.001
  betas: [0.9, 0.999]

# Careful (AdamW with weight decay)
optimizer:
  name: "adamw"
  learning_rate: 0.0001
  weight_decay: 1e-5
```

---

## 🎯 Configuration Templates by Use Case

### Use Case 1: Publication Baseline
```yaml
experiment:
  name: "baseline_for_paper"
  seed: 42                              # Fixed seed

dataset:
  samples_per_class: 50
  train_split: 0.7
  preprocessing:
    normalize: true
    augmentation: false

training:
  epochs: 20
  batch_size: 16
  learning_rate: 0.001
  early_stopping:
    enabled: true
    patience: 10

signal_simulation:
  target_snr_db: 10.0
  photonic_enabled: true

reproducibility:
  set_deterministic: true
```

### Use Case 2: Robustness Testing
```yaml
experiment:
  name: "robustness_test"
  seed: 42

dataset:
  samples_per_class: 100
  preprocessing:
    augmentation: true                  # More variety

training:
  epochs: 50
  batch_size: 8
  early_stopping:
    enabled: true
    patience: 10
  lr_schedule:
    enabled: true

signal_simulation:
  target_snr_db: 0.0                  # Very challenging
  photonic_enabled: true
  laser_linewidth_hz: 1e5             # High phase noise
  temperature_drift_enabled: true
```

### Use Case 3: Hyperparameter Search
```yaml
# Run with multiple parameter variations:

# Quick sweep (3 learning rates)
configs:
  - "config_lr_0001.yaml"
  - "config_lr_001.yaml"
  - "config_lr_01.yaml"

# Batch size sweep (3 sizes)
configs:
  - "config_bs_8.yaml"
  - "config_bs_16.yaml"
  - "config_bs_32.yaml"

# SNR sweep (4 difficulty levels)
configs:
  - "config_snr_15.yaml"   # Easy
  - "config_snr_10.yaml"   # Medium
  - "config_snr_5.yaml"    # Hard
  - "config_snr_0.yaml"    # Very hard
```

### Use Case 4: Real-Time Deployment
```yaml
experiment:
  name: "deployment_model"

training:
  epochs: 20
  batch_size: 16
  learning_rate: 0.001

model:
  rd_branch:
    conv_layers:
      - {out_channels: 32}              # Smaller for speed
      - {out_channels: 64}
    dropout: 0.2

signal_simulation:
  photonic_enabled: true
  target_snr_db: 10.0                  # Realistic conditions
```

---

## 📊 Metrics Interpretation

### Performance Metrics
| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 0-1 | Overall correctness; affected by class imbalance |
| **Precision** | 0-1 | False positive rate; important for deployment |
| **Recall** | 0-1 | Miss rate; important for safety-critical apps |
| **F1-Score** | 0-1 | Harmonic mean; balances precision & recall |

### Radar-Specific Metrics
| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Pd (Detection Probability)** | 0-1 | % of targets correctly detected (want >90%) |
| **FAR (False Alarm Rate)** | 0-1 | % of false detections (want <10%) |
| **ROC-AUC** | 0-1 | Discriminative ability (want >0.95) |

---

## ⚙️ Hardware Considerations

### GPU Requirements
```yaml
# GPU: NVIDIA RTX 3090 (24GB)
batch_size: 32      # Max comfortable
epochs: 100         # ~8 hours

# GPU: NVIDIA RTX 2080Ti (11GB)
batch_size: 16      # Recommended
epochs: 50          # ~4 hours

# GPU: NVIDIA Tesla V100 (16GB)
batch_size: 32      # Comfortable
epochs: 100         # ~6 hours

# CPU only
batch_size: 4       # Reduce to fit memory
epochs: 10          # ~30 min per epoch
```

### Memory Estimation
```
Memory ≈ batch_size × model_size × 2
         (factor of 2 for gradients)

Small model (32→64 CNN):
  16 batch × 100MB × 2 ≈ 3.2 GB

Large model (128→256 CNN):
  16 batch × 500MB × 2 ≈ 16 GB
```

---

## 🔍 Debugging Checklist

### If Model Doesn't Train
- [ ] Reduce `learning_rate` by 10×
- [ ] Check data is being loaded
- [ ] Verify `batch_size` fits in GPU memory
- [ ] Set `verbose: true` in logging

### If Accuracy is Poor
- [ ] Increase `epochs` to 50+
- [ ] Reduce `learning_rate`
- [ ] Enable `early_stopping`
- [ ] Increase `samples_per_class`
- [ ] Test with `photonic_enabled: false`

### If Training is Too Slow
- [ ] Reduce `samples_per_class`
- [ ] Increase `batch_size`
- [ ] Reduce number of `conv_layers`
- [ ] Set `num_workers: 0` for debugging
- [ ] Use smaller model

### If Results Aren't Reproducible
- [ ] Set `random_seed: 42` in experiment
- [ ] Set all `*_seed: 42` in reproducibility
- [ ] Set `set_deterministic: true`
- [ ] Set `benchmark: false`

---

## 📈 Parameter Sweep Examples

### Learning Rate Sweep
```bash
#!/bin/bash
for lr in 0.00001 0.0001 0.001 0.01 0.1; do
  cat > config_lr.yaml << EOF
experiment:
  name: "lr_sweep_$lr"
training:
  learning_rate: $lr
EOF
  python experiment_runner.py --config config_lr.yaml
done
```

### Batch Size Sweep
```bash
for bs in 4 8 16 32 64; do
  sed "s/batch_size: .*/batch_size: $bs/" \
    radar_ai_experiment.yaml > config_bs_$bs.yaml
  python experiment_runner.py --config config_bs_$bs.yaml
done
```

### SNR Difficulty Sweep
```bash
for snr in -5 0 5 10 15 20; do
  sed "s/target_snr_db: .*/target_snr_db: $snr/" \
    radar_ai_experiment.yaml > config_snr_$snr.yaml
  python experiment_runner.py --config config_snr_$snr.yaml
done
```

---

## 💾 File Locations

```
Project Root/
├── radar_ai_experiment.yaml          # Main comprehensive config
├── experiment_config_example.yaml    # Minimal example
├── config.yaml                       # System config
├── RADAR_AI_CONFIG_GUIDE.md          # Detailed guide (this folder)
└── experiments/
    ├── exp_20260220_143022/
    │   ├── models/model_final.pt
    │   ├── reports/metrics.json
    │   └── logs/experiment.log
    └── exp_20260220_150000/
        └── ... (next experiment)
```

---

## 🚀 Quick Commands

```bash
# Run with defaults
python experiment_runner.py --config radar_ai_experiment.yaml

# Run with custom config
python experiment_runner.py --config my_experiment.yaml

# Monitor running experiment
tail -f experiments/exp_*/logs/experiment.log

# View latest results
cat experiments/exp_*/reports/metrics.json | python -m json.tool

# Compare all experiments
for dir in experiments/exp_*/; do
  echo "$(basename $dir): $(grep accuracy $dir/reports/metrics.json)"
done

# Plot training history
python -c "
import json
with open('experiments/exp_*/reports/training_history.json') as f:
    h = json.load(f)
    import matplotlib.pyplot as plt
    plt.plot(h['epoch'], h['loss'])
    plt.savefig('loss.png')
"
```

---

## 📚 Parameter Cheat Sheet

**Fastest**: 10 epochs, 32 batch, 0.01 lr, 25 samples/class  
**Balanced**: 20 epochs, 16 batch, 0.001 lr, 50 samples/class  
**Best**: 50 epochs, 8 batch, 0.0001 lr, 100 samples/class  

**Easy Problem**: SNR 15 dB, no photonic  
**Realistic**: SNR 10 dB, photonic ON  
**Hard**: SNR 0-5 dB, photonic + thermal drift  

---

For detailed information, see **RADAR_AI_CONFIG_GUIDE.md**


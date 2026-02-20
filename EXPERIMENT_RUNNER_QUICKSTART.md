# Experiment Runner Quick Start

## 5-Minute Setup

### 1. Basic Run (Default Config)
```bash
cd /home/nikhil/PycharmProjects/"Aegis Cognitive Defense Platform"
python experiment_runner.py
```

### 2. Custom Config
```bash
python experiment_runner.py --config experiment_config_example.yaml
```

## Expected Output

```
[SEEDS] Set random seed to 42
[DIRS] Created directory: outputs/exp_20260220_143022/models
[DIRS] Created directory: outputs/exp_20260220_143022/logs
[DIRS] Created directory: outputs/exp_20260220_143022/plots
[DIRS] Created directory: outputs/exp_20260220_143022/reports

==================================================
STAGE 1: DATA PREPROCESSING
==================================================
2026-02-20 14:30:22 - experiment - INFO - Creating dataset with 50 samples per class
2026-02-20 14:30:22 - experiment - INFO - Range-Doppler shape: (300, 128, 128)
2026-02-20 14:30:22 - experiment - INFO - Spectrogram shape: (300, 128, 128)
2026-02-20 14:30:22 - experiment - INFO - Metadata shape: (300, 8)
2026-02-20 14:30:22 - experiment - INFO - Labels shape: (300,)

==================================================
STAGE 2: MODEL TRAINING
==================================================
2026-02-20 14:30:23 - experiment - INFO - Model architecture: PhotonicRadarAI
2026-02-20 14:30:23 - experiment - INFO - Training epochs: 20
2026-02-20 14:30:23 - experiment - INFO - Batch size: 16
2026-02-20 14:30:23 - experiment - INFO - Learning rate: 0.001
Epoch   1/20 - Loss: 1.8234
Epoch   5/20 - Loss: 1.3421
Epoch  10/20 - Loss: 0.9876
Epoch  15/20 - Loss: 0.6543
Epoch  20/20 - Loss: 0.4321
2026-02-20 14:30:45 - experiment - INFO - Training completed

==================================================
STAGE 3: MODEL EVALUATION
==================================================
2026-02-20 14:30:46 - experiment - INFO - Accuracy: 0.9167
2026-02-20 14:30:46 - experiment - INFO - Probability of Detection (Pd): 0.9167
2026-02-20 14:30:46 - experiment - INFO - False Alarm Rate (FAR): 0.0833
2026-02-20 14:30:46 - experiment - INFO - Saved confusion matrix to: outputs/exp_20260220_143022/plots/confusion_matrix.png

==================================================
STAGE 4: SAVING RESULTS
==================================================
2026-02-20 14:30:46 - experiment - INFO - Saved model to: outputs/exp_20260220_143022/models/model_final.pt
2026-02-20 14:30:46 - experiment - INFO - Saved metrics to: outputs/exp_20260220_143022/reports/metrics.json
2026-02-20 14:30:46 - experiment - INFO - Saved training history to: outputs/exp_20260220_143022/reports/training_history.json
2026-02-20 14:30:46 - experiment - INFO - Saved config to: outputs/exp_20260220_143022/reports/config.yaml

==================================================
EXPERIMENT SUMMARY
==================================================
Start time: 2026-02-20 14:30:22
End time: 2026-02-20 14:30:46
Total duration: 0:00:24.123456

Performance Metrics:
  Accuracy: 0.9167
  Probability of Detection: 0.9167
  False Alarm Rate: 0.0833

Output directory: outputs/exp_20260220_143022/
==================================================
```

## View Results

After experiment completes:

### Check Metrics
```bash
cat outputs/exp_*/reports/metrics.json | python -m json.tool
```

### View Log
```bash
tail -f outputs/exp_*/logs/experiment.log
```

### List All Experiments
```bash
ls -la outputs/
```

### Load Trained Model
```python
import torch
from src.model_pytorch import build_pytorch_model

# Load model
model = build_pytorch_model(num_classes=6)
state = torch.load("outputs/exp_20260220_143022/models/model_final.pt")
model.load_state_dict(state)
model.eval()

# Use for inference
with torch.no_grad():
    prediction = model(rd_tensor, spec_tensor, meta_tensor)
```

## Configuration Reference

### Minimal Config
```yaml
experiment:
  seed: 42
  samples_per_class: 50

training:
  epochs: 20
  batch_size: 16
  learning_rate: 0.001
```

### Full Config
```yaml
experiment:
  name: "my_exp"
  description: "My experiment"
  seed: 42
  output_dir: outputs
  samples_per_class: 50

training:
  epochs: 20
  batch_size: 16
  learning_rate: 0.001
  validation_split: 0.2

model_config:
  num_classes: 6
  metadata_size: 8
  input_height: 128
  input_width: 128

logging:
  level: INFO
  dir: results
  file: system.log
  max_bytes: 10485760
  backup_count: 5
```

## Common Commands

```bash
# Run with default config
python experiment_runner.py

# Run with custom config
python experiment_runner.py --config my_config.yaml

# Get help
python experiment_runner.py --help

# Run in background and save output
nohup python experiment_runner.py > experiment.log 2>&1 &

# Monitor running experiment
tail -f outputs/exp_*/logs/experiment.log

# Find latest experiment
ls -t outputs/ | head -1
```

## Experiment Directory Structure

```
outputs/exp_20260220_143022/
├── models/
│   └── model_final.pt              # Trained PyTorch model
├── logs/
│   └── experiment.log              # Full experiment log
├── plots/
│   └── confusion_matrix.png        # Confusion matrix visualization
└── reports/
    ├── metrics.json                # Final evaluation metrics
    ├── training_history.json       # Per-epoch training loss
    └── config.yaml                 # Configuration copy
```

## Troubleshooting

### Script not found
```bash
cd /home/nikhil/PycharmProjects/"Aegis Cognitive Defense Platform"
python experiment_runner.py
```

### Permission denied
```bash
chmod +x experiment_runner.py
./experiment_runner.py
```

### Out of memory
Edit `experiment_config_example.yaml`:
```yaml
training:
  batch_size: 8           # Reduce from 16
  
experiment:
  samples_per_class: 25   # Reduce from 50
```

### GPU not detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If False, modify config to use CPU (no changes needed, CPU is default).

## Next Steps

1. ✅ Run default experiment: `python experiment_runner.py`
2. ✅ View results in `outputs/` directory
3. ✅ Create custom config for your hyperparameters
4. ✅ Run multiple experiments for comparison
5. ✅ Load trained model for inference

For more details, see [EXPERIMENT_RUNNER.md](EXPERIMENT_RUNNER.md)

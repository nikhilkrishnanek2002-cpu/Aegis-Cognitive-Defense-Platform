# Experiment Runner Integration Guide

## What is the Experiment Runner?

The `experiment_runner.py` is a production-ready orchestrator that automates the complete ML pipeline for your Cognitive Radar AI project:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT RUNNER PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

INPUT: YAML Configuration
│
├─ experiment_config_example.yaml
│  └─ Defines seeds, hyperparameters, output paths
│
↓

STAGE 1: DATA PREPROCESSING
│
├─ Call: create_pytorch_dataset(samples_per_class)
├─ Input: Configuration for signal generation
├─ Output: 
│  ├─ Range-Doppler maps (128×128)
│  ├─ Spectrograms (128×128)
│  ├─ Photonic metadata (8-dim)
│  └─ Class labels (6 classes)
│
↓

STAGE 2: MODEL TRAINING
│
├─ Call: build_pytorch_model() + Adam optimizer
├─ Input: Preprocessed data tensors
├─ Process:
│  ├─ Initialize PhotonicRadarAI model
│  ├─ Setup DataLoader (batch_size=16)
│  ├─ Loop: epochs=20
│  │  └─ Compute loss, backward, optimize
│  └─ Log per-epoch training loss
├─ Output: Trained model state_dict
│
↓

STAGE 3: MODEL EVALUATION
│
├─ Call: evaluate_pytorch()
├─ Input: Trained model + test data
├─ Output:
│  ├─ Accuracy: 91.67%
│  ├─ Probability of Detection: 91.67%
│  ├─ False Alarm Rate: 8.33%
│  └─ Confusion matrix heatmap (plot)
│
↓

STAGE 4: RESULTS PERSISTENCE
│
├─ Save: model_final.pt (trained weights)
├─ Save: metrics.json (accuracy, Pd, FAR, confusion matrix)
├─ Save: training_history.json (per-epoch loss)
├─ Save: config.yaml (configuration copy)
└─ Save: experiment.log (detailed log)

OUTPUT: Organized experiment directory
│
outputs/exp_20260220_143022/
├── models/model_final.pt
├── logs/experiment.log
├── plots/confusion_matrix.png
└── reports/
    ├── metrics.json
    ├── training_history.json
    └── config.yaml
```

## Integration Points with Existing Modules

| Module | Function | Usage |
|--------|----------|-------|
| `src/train_pytorch.py` | `create_pytorch_dataset()` | **Data generation** in Stage 1 |
| `src/model_pytorch.py` | `build_pytorch_model()` | **Model initialization** in Stage 2 |
| `src/evaluate.py` | `evaluate_pytorch()` | **Metrics computation** in Stage 3 |
| `src/logger.py` | `init_logging()` | **Logging setup** (optional) |
| PyTorch native | DataLoader, optimizers, losses | **Training loop** in Stage 2 |

## Clean Modular Calls

### Data Preprocessing
```python
from src.train_pytorch import create_pytorch_dataset

# Clean single call - data is generated and preprocessed
rd, spec, meta, y = create_pytorch_dataset(samples_per_class=50)

# Returns:
# - rd: Range-Doppler maps (300, 128, 128)
# - spec: Spectrograms (300, 128, 128)
# - meta: Metadata features (300, 8)
# - y: Class labels (300,)
```

### Model Training
```python
from src.model_pytorch import build_pytorch_model
import torch.optim as optim
import torch.nn as nn

# Initialize model
model = build_pytorch_model(num_classes=6)

# Clean manual training loop
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    for rd, spec, meta, y in loader:
        outputs = model(rd, spec, meta)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
```

### Model Evaluation
```python
from sklearn.metrics import confusion_matrix, accuracy_score

# Inference in evaluation mode
model.eval()
with torch.no_grad():
    outputs = model(rd, spec, meta)
    preds = torch.argmax(outputs, dim=1)

# Compute metrics
accuracy = accuracy_score(y_true, preds)
cm = confusion_matrix(y_true, preds)
Pd = np.trace(cm) / np.sum(cm)
FAR = (np.sum(cm) - np.trace(cm)) / np.sum(cm)
```

## Configuration Handling

### YAML Structure
```yaml
experiment:        # Experiment metadata and reproducibility
  seed: 42
  output_dir: outputs

training:          # Training hyperparameters
  epochs: 20
  batch_size: 16
  learning_rate: 0.001

model_config:      # Model architecture parameters
  num_classes: 6
  metadata_size: 8

logging:           # Logging configuration
  level: INFO
  dir: results
```

### Seed Management
```python
def _set_seeds(self, seed: int = None) -> None:
    """Set reproducible random state"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

Result: Same seed → Same random data → Same results ✓

## Output Organization

### Automatic Directory Creation
```python
def _create_output_dirs(self) -> Dict[str, str]:
    """Creates timestamped experiment directories"""
    base_output = config.get('experiment', {}).get('output_dir', 'outputs')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_output, f"exp_{timestamp}")
    
    # Creates:
    # outputs/exp_20260220_143022/
    # ├── models/
    # ├── logs/
    # ├── plots/
    # └── reports/
```

### Results Files

#### `metrics.json` - Final Performance
```json
{
  "accuracy": 0.9167,
  "probability_of_detection": 0.9167,
  "false_alarm_rate": 0.0833,
  "confusion_matrix": [
    [50, 0, 0, 0, 0, 0],
    [2, 48, 0, 0, 0, 0],
    ...
  ]
}
```

#### `training_history.json` - Per-Epoch Loss
```json
{
  "epoch": [1, 2, 3, ..., 20],
  "loss": [1.8234, 1.6543, 1.4321, ..., 0.4321]
}
```

#### `experiment.log` - Complete Audit Trail
```
2026-02-20 14:30:22 - experiment - INFO - Experiment started at 2026-02-20 14:30:22.123456
2026-02-20 14:30:22 - experiment - INFO - Config loaded from: config.yaml
2026-02-20 14:30:22 - experiment - INFO - Set random seed to 42
2026-02-20 14:30:22 - experiment - INFO - Created directory: outputs/exp_20260220_143022/models
...
```

## Logging Strategy

### Dual-Channel Logging
```python
def _setup_logging(self) -> logging.Logger:
    """Logs to both console and file"""
    
    # Console: Real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    
    # File: Persistent record
    file_handler = logging.FileHandler(log_file)
    
    # Both use same format
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
```

Benefits:
- ✓ Real-time progress in terminal
- ✓ Complete log saved for review
- ✓ Easy debugging and reproducibility

## GPU/CPU Support

### Automatic Detection
```python
def _setup_device(self) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # GPU detected and used
    else:
        device = torch.device('cpu')
        # CPU used as fallback
```

Behavior:
- GPU detected → Uses CUDA automatically ✓
- GPU not available → Falls back to CPU gracefully ✓
- No configuration needed ✓

## Class Architecture

### ExperimentRunner class
Orchestrates the complete pipeline with methods:

```
ExperimentRunner
├── __init__(config_path)
│   └─ Load YAML config
│
├── _set_seeds()
│   └─ Set global RNG reproducibility
│
├── _create_output_dirs()
│   └─ Create timestamped directory structure
│
├── _setup_logging()
│   └─ Initialize console + file logging
│
├── _setup_device()
│   └─ Detect GPU/CPU availability
│
├── _preprocess_data()
│   └─ Call create_pytorch_dataset()
│
├── _train_model()
│   └─ Call build_pytorch_model() + training loop
│
├── _evaluate_model()
│   └─ Compute metrics + create visualizations
│
├── _save_results()
│   └─ Persist models and metrics
│
├── _print_summary()
│   └─ Print experiment summary
│
└── run()
    └─ Execute full pipeline (main entry point)
```

## Usage Patterns

### Pattern 1: One-Off Experiment
```bash
python experiment_runner.py --config config.yaml
# Results → outputs/exp_20260220_143022/
```

### Pattern 2: Hyperparameter Sweep
```bash
for lr in 0.0001 0.001 0.01; do
  echo "learning_rate: $lr" > config_lr.yaml
  python experiment_runner.py --config config_lr.yaml
done
# Results → outputs/exp_*/ (multiple timestamped dirs)
```

### Pattern 3: Batch Processing
```bash
python experiment_runner.py --config config1.yaml
python experiment_runner.py --config config2.yaml
python experiment_runner.py --config config3.yaml
# Results → outputs/exp_1/, outputs/exp_2/, outputs/exp_3/
```

### Pattern 4: Background Execution
```bash
nohup python experiment_runner.py > experiment.log 2>&1 &
# Monitor with:
tail -f outputs/exp_*/logs/experiment.log
```

## Error Handling & Robustness

### Graceful Error Handling
```python
def run(self) -> None:
    try:
        # Pipeline stages
        ...
    except Exception as e:
        if self.logger:
            self.logger.exception(f"Failed: {str(e)}")
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
```

### File Validation
```python
def _load_config(self) -> Dict[str, Any]:
    if not os.path.exists(self.config_path):
        raise FileNotFoundError(f"Config not found: {self.config_path}")
    # Load and return
```

## Performance Characteristics

| Stage | Time | Memory |
|-------|------|--------|
| Preprocessing | ~5-10s | GPU: 100MB |
| Training (20 epochs, bs=16) | ~15-25s | GPU: 2-4GB |
| Evaluation | ~2-5s | GPU: 100MB |
| Total | ~30-45s | Peak: 2-4GB |

*Timings estimated on RTX 3090 GPU. CPU will be 5-10× slower.*

## Reproducibility Guarantee

With fixed seed:
```python
experiment:
  seed: 42  # Fixed
```

You get:
- ✓ Identical random data generation
- ✓ Identical model initialization
- ✓ Identical training trajectory
- ✓ Identical final metrics

Perfect for:
- Publishing results
- Comparing hyperparameters
- Debugging issues
- Validating implementations


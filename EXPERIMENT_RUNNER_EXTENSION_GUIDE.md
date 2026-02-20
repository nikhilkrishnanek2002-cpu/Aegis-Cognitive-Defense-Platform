# Experiment Runner - Developer Extension Guide

## Extending the Experiment Runner

This guide explains how to customize and extend the experiment runner for your specific needs.

---

## Architecture for Extension

```
ExperimentRunner (Base Class)
│
├─ Configuration Layer
│  └─ _load_config() - Load/parse YAML
│
├─ Setup Layer
│  ├─ _set_seeds() - Reproducibility
│  ├─ _create_output_dirs() - Organization
│  ├─ _setup_logging() - Audit trail
│  └─ _setup_device() - Hardware detection
│
├─ Pipeline Stages (Extensible)
│  ├─ _preprocess_data() - Can add new preprocessing
│  ├─ _train_model() - Can swap model/optimizer
│  ├─ _evaluate_model() - Can add new metrics
│  └─ _save_results() - Can add new artifacts
│
└─ Utilities
   ├─ _print_summary() - Report results
   └─ run() - Orchestrate all stages
```

---

## Common Extensions

### Extension 1: Use Different Model

**Problem**: Want to use PyTorch model instead of current one

**Solution**:
```python
# Modify _train_model():

def _train_model(self, rd, spec, meta, y):
    # ... existing code ...
    
    # Change this:
    # model = build_pytorch_model(num_classes=num_classes)
    
    # To this:
    from my_custom_models import MyAdvancedModel
    model = MyAdvancedModel(num_classes=num_classes)
    
    # Rest of training loop stays the same
```

### Extension 2: Add Custom Metrics

**Problem**: Need additional evaluation metrics

**Solution**:
```python
# Add to _evaluate_model():

def _evaluate_model(self, model, rd, spec, meta, y):
    # ... existing code to get predictions ...
    
    # Existing metrics
    accuracy = accuracy_score(all_y, all_preds)
    
    # Add new metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_y, all_preds, average='weighted')
    recall = recall_score(all_y, all_preds, average='weighted')
    f1 = f1_score(all_y, all_preds, average='weighted')
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        # ... existing metrics ...
    }
    
    return metrics
```

### Extension 3: Add Custom Preprocessing

**Problem**: Need specialized data preprocessing

**Solution**:
```python
# Add new method:

def _custom_preprocessing(self, rd, spec, meta, y):
    """Apply custom preprocessing to data"""
    self.logger.info("Applying custom preprocessing...")
    
    # Your preprocessing logic
    rd_processed = self._normalize_advanced(rd)
    spec_processed = self._apply_augmentation(spec)
    
    return rd_processed, spec_processed, meta, y

# Modify _preprocess_data():

def _preprocess_data(self):
    self.logger.info("=" * 50)
    self.logger.info("STAGE 1: DATA PREPROCESSING")
    self.logger.info("=" * 50)
    
    samples_per_class = self.config.get('experiment', {}).get('samples_per_class', 50)
    rd, spec, meta, y = create_pytorch_dataset(samples_per_class=samples_per_class)
    
    # Add custom preprocessing
    rd, spec, meta, y = self._custom_preprocessing(rd, spec, meta, y)
    
    self.logger.info(f"Range-Doppler shape: {rd.shape}")
    # ... rest of method ...
    
    return rd, spec, meta, y
```

### Extension 4: Add New Configuration Section

**Problem**: Need custom config parameters

**Solution**:
```yaml
# In config.yaml, add:

experiment:
  seed: 42
  output_dir: outputs

custom_feature:
  enabled: true
  param1: value1
  param2: value2

training:
  # ... existing ...
```

```python
# In experiment_runner.py, access it:

def _preprocess_data(self):
    custom_config = self.config.get('custom_feature', {})
    param1 = custom_config.get('param1', 'default')
    param2 = custom_config.get('param2', 'default')
    
    self.logger.info(f"Custom feature param1: {param1}")
    self.logger.info(f"Custom feature param2: {param2}")
```

### Extension 5: Add New Visualization

**Problem**: Need custom plots beyond confusion matrix

**Solution**:
```python
# Add method:

def _plot_training_loss(self, history):
    """Plot training loss over time"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['loss'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plot_path = os.path.join(self.output_dirs['plots'], 'training_loss.png')
    plt.savefig(plot_path)
    plt.close()
    
    self.logger.info(f"Saved training loss plot to: {plot_path}")

# Call from _save_results():

def _save_results(self, model, history, metrics):
    # ... existing code ...
    
    # Add new visualization
    self._plot_training_loss(history)
    
    # ... rest of save_results ...
```

### Extension 6: Early Stopping

**Problem**: Want to stop training if loss doesn't improve

**Solution**:
```python
# Add to _train_model():

def _train_model(self, rd, spec, meta, y):
    # ... existing setup code ...
    
    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0
    
    history = {'loss': [], 'epoch': []}
    
    for epoch in range(epochs):
        # ... training code ...
        
        avg_loss = running_loss / batch_count
        history['loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            self.logger.info(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {avg_loss:.4f}")
    
    return model, history
```

### Extension 7: Model Checkpointing

**Problem**: Save best model during training

**Solution**:
```python
# Add to _train_model():

def _train_model(self, rd, spec, meta, y):
    # ... existing setup code ...
    
    best_loss = float('inf')
    best_model_path = os.path.join(self.output_dirs['models'], 'best_model.pt')
    
    for epoch in range(epochs):
        # ... training code ...
        
        avg_loss = running_loss / batch_count
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            self.logger.info(f"Saved best model with loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            self.logger.info(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {avg_loss:.4f}")
    
    return model, history
```

### Extension 8: Validation Set Monitoring

**Problem**: Track performance on validation set

**Solution**:
```python
# Add to _train_model():

def _train_model(self, rd, spec, meta, y):
    # ... existing setup code ...
    
    # Split into train/val
    val_split = self.config.get('training', {}).get('validation_split', 0.2)
    n_val = int(len(rd) * val_split)
    
    indices = torch.randperm(len(rd))
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_dataset = TensorDataset(
        rd[train_indices], spec[train_indices], 
        meta[train_indices], y[train_indices]
    )
    val_dataset = TensorDataset(
        rd[val_indices], spec[val_indices], 
        meta[val_indices], y[val_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {'train_loss': [], 'val_loss': [], 'epoch': []}
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for b_rd, b_spec, b_meta, b_y in train_loader:
            # ... training step ...
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_rd, b_spec, b_meta, b_y in val_loader:
                # ... validation step ...
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)
        
        self.logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    return model, history
```

### Extension 9: Experiment Tracking (MLflow)

**Problem**: Need experiment tracking and comparison

**Solution**:
```python
# Install: pip install mlflow

def _setup_mlflow(self):
    """Setup MLflow tracking"""
    import mlflow
    
    mlflow.set_experiment("CognitiveRadarAI")
    mlflow.start_run()
    
    # Log config parameters
    for key, value in self.config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                mlflow.log_param(f"{key}_{sub_key}", sub_value)
        else:
            mlflow.log_param(key, value)

def run(self):
    # ... existing setup ...
    
    self._setup_mlflow()
    
    # ... pipeline stages ...
    
    # Log metrics
    import mlflow
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            mlflow.log_metric(key, value)
    
    mlflow.end_run()
```

### Extension 10: Distributed Training

**Problem**: Need multi-GPU training

**Solution**:
```python
# Install: pip install torch-distributed-launch

def _train_model_distributed(self, rd, spec, meta, y):
    """Train with DataParallel on multiple GPUs"""
    import torch.nn as nn
    from torch.nn.parallel import DataParallel
    
    # ... existing setup ...
    
    model = build_pytorch_model(num_classes=num_classes)
    
    # Use multiple GPUs
    if torch.cuda.device_count() > 1:
        self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model = model.to(self.device)
    
    # ... rest of training loop ...
```

---

## Testing Your Extensions

### Unit Test Template
```python
# tests/test_custom_extension.py

import pytest
import torch
import yaml
from experiment_runner import ExperimentRunner


class TestCustomExtension:
    
    @pytest.fixture
    def runner(self):
        config_path = 'test_config.yaml'
        return ExperimentRunner(config_path)
    
    def test_custom_preprocessing(self, runner):
        """Test custom preprocessing function"""
        rd = torch.randn(10, 128, 128)
        spec = torch.randn(10, 128, 128)
        meta = torch.randn(10, 8)
        y = torch.randint(0, 6, (10,))
        
        rd_processed, spec_processed, meta_processed, y_processed = runner._custom_preprocessing(
            rd, spec, meta, y
        )
        
        assert rd_processed.shape == rd.shape
        assert spec_processed.shape == spec.shape
        assert y_processed.shape == y.shape
    
    def test_custom_metrics(self, runner):
        """Test custom metrics computation"""
        model = build_pytorch_model()
        model.eval()
        
        all_y = torch.randint(0, 6, (100,)).numpy()
        all_preds = torch.randint(0, 6, (100,)).numpy()
        
        # This should not raise
        metrics = runner._compute_custom_metrics(all_y, all_preds)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

# Run tests:
# pytest tests/test_custom_extension.py -v
```

---

## Integration Points

### Configuration
**File**: `experiment_config_example.yaml` or `config.yaml`

**Adding new config section**:
```yaml
my_feature:
  param1: value1
  param2: value2
```

**Accessing in code**:
```python
config = self.config.get('my_feature', {})
param1 = config.get('param1', 'default')
```

### Logging
**Method**: `self.logger`

**Available in**: Any method of `ExperimentRunner`

**Usage**:
```python
self.logger.info("Information message")
self.logger.warning("Warning message")
self.logger.error("Error message")
self.logger.debug("Debug message")
```

### Output Paths
**Available**: `self.output_dirs` (dict)

**Keys**:
- `base`: Base experiment directory
- `models`: Model save location
- `logs`: Log file location
- `plots`: Plot save location
- `reports`: Report save location

**Usage**:
```python
model_path = os.path.join(self.output_dirs['models'], 'my_model.pt')
plot_path = os.path.join(self.output_dirs['plots'], 'my_plot.png')
report_path = os.path.join(self.output_dirs['reports'], 'results.json')
```

### Device
**Available**: `self.device` (torch.device)

**Usage**:
```python
tensor = tensor.to(self.device)
model = model.to(self.device)
```

---

## Checklist for Adding Extensions

- [ ] Identify extension type (metric, model, visualization, etc.)
- [ ] Choose where to add in pipeline (which stage)
- [ ] Add config section if needed to `config.yaml`
- [ ] Implement extension method in `ExperimentRunner`
- [ ] Call new method from appropriate pipeline stage
- [ ] Add logging statements for debugging
- [ ] Save outputs to `self.output_dirs`
- [ ] Update documentation if needed
- [ ] Write unit tests
- [ ] Test with sample config and data
- [ ] Verify output files are created correctly

---

## Common Pitfalls

### Pitfall 1: Not Moving Tensors to Device
```python
# Wrong
outputs = model(rd, spec, meta)

# Right
outputs = model(
    rd.to(self.device), 
    spec.to(self.device), 
    meta.to(self.device)
)
```

### Pitfall 2: Not Checking Config Existence
```python
# Wrong
param = self.config['my_section']['my_param']  # KeyError if missing

# Right
param = self.config.get('my_section', {}).get('my_param', 'default')
```

### Pitfall 3: Forgetting Evaluation Mode
```python
# Wrong
with torch.no_grad():
    outputs = model(x)  # model still in training mode!

# Right
model.eval()
with torch.no_grad():
    outputs = model(x)
```

### Pitfall 4: Not Handling Batch Dimensions
```python
# Input from DataLoader: (batch_size, channels, height, width)
# Sometimes you'll get: (batch_size, height, width)

# Handle both cases:
if x.dim() == 3:
    x = x.unsqueeze(1)  # Add channel dimension
```

### Pitfall 5: Writing to Wrong Output Directory
```python
# Wrong
plt.savefig('confusion_matrix.png')

# Right
path = os.path.join(self.output_dirs['plots'], 'confusion_matrix.png')
plt.savefig(path)
```

---

## Performance Optimization Tips

### 1. DataLoader Optimization
```python
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,           # Use multiprocessing
    pin_memory=True,         # Pre-allocate GPU memory
    persistent_workers=True  # Keep workers alive
)
```

### 2. Model Optimization
```python
# Use half precision (FP16) training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(x)
    loss = criterion(outputs, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Gradient Checkpointing
```python
# For very large models, save memory
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

---

## Debugging Tips

### Print Model Structure
```python
from torchsummary import summary
summary(model, input_size=[(1, 128, 128), (1, 128, 128), (8,)])
```

### Print Intermediate Activations
```python
class DebugModel(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        print(f"After layer1: {x.shape}")  # Debug
        x = self.layer2(x)
        print(f"After layer2: {x.shape}")  # Debug
        return x
```

### Check Data Distribution
```python
import numpy as np
print(f"Mean: {rd.mean():.4f}")
print(f"Std: {rd.std():.4f}")
print(f"Min: {rd.min():.4f}")
print(f"Max: {rd.max():.4f}")
```

---

## References

- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/


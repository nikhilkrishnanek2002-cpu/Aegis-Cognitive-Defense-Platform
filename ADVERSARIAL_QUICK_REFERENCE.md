# Adversarial Radar Attacks - Quick Reference

## Module Overview

**Location:** `src/adversarial_attacks.py` (26 KB)

**Purpose:** Simulate adversarial radar signal attacks for model robustness evaluation

**Status:** ✅ Production-ready, tested, documented

---

## Five Core Attack Functions

### 1. Gaussian Noise Attack
```python
from src.adversarial_attacks import add_gaussian_noise

noisy = add_gaussian_noise(signal, snr_db=10)
# SNR: 5, 10, 15, 20 dB (typical range)
```
**Physical:** AWGN channel | **Effect:** SNR degradation

### 2. Frequency Shift Attack
```python
from src.adversarial_attacks import frequency_shift_attack

shifted = frequency_shift_attack(signal, shift_hz=500, sample_rate=1e6)
# Shift: ±100 to ±5000 Hz (typical range)
```
**Physical:** Doppler effect | **Effect:** Velocity confusion

### 3. Amplitude Scaling Attack
```python
from src.adversarial_attacks import amplitude_scaling_attack

scaled = amplitude_scaling_attack(signal, scale=0.5)
# Scale: 0.3 to 1.5 (typical range)
```
**Physical:** Power manipulation | **Effect:** Range ambiguity

### 4. Replay Attack
```python
from src.adversarial_attacks import replay_attack

echoed = replay_attack(signal, delay_samples=100)
# Delay: 50 to 500 samples (typical range)
```
**Physical:** Multipath echo | **Effect:** Ghost target

### 5. Spoofing Attack
```python
from src.adversarial_attacks import spoof_target_attack

spoofed = spoof_target_attack(
    signal, 
    fake_range=500,      # meters
    fake_velocity=30,    # m/s
    carrier_freq=10e9,   # Hz
    sample_rate=1e6      # Hz
)
```
**Physical:** Coherent injection | **Effect:** False detection

---

## Attack Suite Orchestration

### Full Evaluation
```python
from src.adversarial_attacks import run_attack_suite
import torch

results = run_attack_suite(
    model=model,
    signals=test_signals,           # (n_samples, n_features)
    labels=test_labels,             # (n_samples,)
    attack_params={
        'noise_snr_db': [5, 10, 15, 20],
        'freq_shift_hz': [100, 500],
        'amplitude_scales': [0.5, 1.2],
        'replay_delays_samples': [100, 500],
        'spoof_ranges': [100, 500],
        'spoof_velocities': [10, 50],
    },
    device='cuda',  # or 'cpu'
    batch_size=64
)
```

### Results Structure
```python
results = {
    'baseline_accuracy': 0.95,          # Clean signals
    'baseline_conf_matrix': [...],      # Confusion matrix
    'attacks': {
        'gaussian_noise': {
            'snr_5_db': {'accuracy': ..., 'confusion_matrix': ...},
            'snr_10_db': {...},
            ...
            'mean_accuracy': 0.92,
            'min_accuracy': 0.90,
        },
        'frequency_shift': {...},
        'amplitude_scaling': {...},
        'replay': {...},
        'spoof': {...},
    },
    'summary': {
        'baseline_accuracy': 0.95,
        'mean_attacked_accuracy': 0.87,
        'min_attacked_accuracy': 0.82,
        'accuracy_drop_mean': 0.08,
        'robustness_score': 0.916,          # attacked/baseline
        ...
    },
    'execution_time': 125.43  # seconds
}
```

---

## Common Usage Patterns

### Pattern 1: Quick Single Attack
```python
# Fast test with one attack type
attacked_signal = add_gaussian_noise(signal, snr_db=10)
prediction = model(torch.FloatTensor(attacked_signal))
confidence = torch.softmax(prediction, dim=-1).max().item()
```

### Pattern 2: Batch Attack Processing
```python
attacks = AdversarialRadarAttacks()
attacked_batch = np.array([
    attacks.add_gaussian_noise(sig, 10) for sig in batch_signals
])
batch_tensor = torch.FloatTensor(attacked_batch)
predictions = model(batch_tensor)
```

### Pattern 3: Full Robustness Report
```python
results = run_attack_suite(model, signals, labels)
print(f"Model robustness: {results['summary']['robustness_score']:.1%}")
print(f"Accuracy drop: {results['summary']['accuracy_drop_mean']:.1%}")

# Save report
import json
with open('robustness_report.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Pattern 4: Vulnerability Analysis
```python
# Find most damaging attack
worst_attack = min(results['attacks'].items(),
    key=lambda x: x[1]['mean_accuracy'])
print(f"Most dangerous: {worst_attack[0]}")
print(f"Causes {worst_attack[1]['mean_accuracy']:.1%} accuracy")
```

### Pattern 5: Custom Parameter Sweep
```python
# Test specific noise levels
results = run_attack_suite(
    model, signals, labels,
    attack_params={
        'noise_snr_db': np.arange(0, 25, 2),  # 0-24 dB
    }
)

# Plot results
snrs = sorted([int(k.split('_')[1]) 
    for k in results['attacks']['gaussian_noise'].keys()])
accs = [results['attacks']['gaussian_noise'][f'snr_{s}_db']['accuracy'] 
    for s in snrs]

import matplotlib.pyplot as plt
plt.plot(snrs, accs, 'o-')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')
plt.show()
```

### Pattern 6: Adversarial Training
```python
# Train model to be robust against attacks
from src.adversarial_attacks import AdversarialRadarAttacks

attacks = AdversarialRadarAttacks()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_signals, batch_labels in train_loader:
        # Random attack
        if np.random.rand() > 0.5:
            batch_signals = np.array([
                attacks.add_gaussian_noise(s, np.random.choice([5,10,15]))
                for s in batch_signals
            ])
        
        outputs = model(torch.FloatTensor(batch_signals))
        loss = loss_fn(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Pattern 7: Model Comparison
```python
# Compare robustness across versions
models = {
    'baseline': torch.load('models/baseline.pt'),
    'hardened': torch.load('models/hardened.pt'),
    'quantized': torch.load('models/quantized.pt'),
}

robustness_scores = {}
for name, model in models.items():
    results = run_attack_suite(model, signals, labels)
    robustness_scores[name] = results['summary']['robustness_score']

# Rank by robustness
for name in sorted(robustness_scores, 
    key=robustness_scores.get, reverse=True):
    print(f"{name:15s}: {robustness_scores[name]:.3f}")
```

---

## Output Formats

### Attack Suite Results (JSON)
```json
{
  "baseline_accuracy": 0.95,
  "attacks": {
    "gaussian_noise": {
      "snr_5_db": {"accuracy": 0.92, "confusion_matrix": [...]},
      "mean_accuracy": 0.92
    }
  },
  "summary": {
    "baseline_accuracy": 0.95,
    "mean_attacked_accuracy": 0.87,
    "robustness_score": 0.916
  },
  "execution_time": 125.43
}
```

### Metrics Returned
| Metric | Meaning |
|--------|---------|
| `baseline_accuracy` | Accuracy on clean signals |
| `mean_attacked_accuracy` | Average accuracy across all attacks |
| `accuracy_drop_mean` | Average accuracy degradation |
| `robustness_score` | attacked_acc / baseline_acc (higher is better) |
| `confusion_matrix` | Classification breakdown per attack |

---

## Performance Characteristics

| Task | Time | Memory |
|------|------|--------|
| Single attack (1000 samples) | <1 ms | <10 KB |
| Batch of 50 signals | ~50 ms | ~500 KB |
| Full suite (50 signals, 5 attacks) | ~100 ms | ~5 MB |
| Full suite (1K signals, 5 attacks) | ~2 s | ~100 MB |
| Full suite (10K signals, 5 attacks) | ~20 s | ~1 GB |

**Tips:**
- Use GPU for >1K signals (5-10x speedup)
- Batch size 64-128 is optimal
- Memory~8 KB per I/Q signal

---

## Typical Attack Parameter Ranges

```python
# Gaussian Noise
'noise_snr_db': [0, 5, 10, 15, 20, 25]      # dB

# Frequency Shift (Doppler equivalents)
'freq_shift_hz': [100, 200, 500, 1000]      # Hz

# Amplitude Scaling
'amplitude_scales': [0.3, 0.5, 0.8, 1.2, 1.5]

# Replay Delay
'replay_delays_samples': [50, 100, 200, 500]

# Spoofing Ranges
'spoof_ranges': [100, 200, 500, 1000]       # meters

# Spoofing Velocities
'spoof_velocities': [5, 10, 20, 50]         # m/s
```

---

## Real-World Conversions

```python
# Doppler frequency from velocity (10 GHz radar)
velocity_ms = 30  # m/s
doppler_hz = 2 * 10e9 * velocity_ms / 3e8  # ≈ 2 kHz

# Range delay from delay samples
delay_samples = 100
sample_rate = 1e6  # Hz
delay_time_us = delay_samples / sample_rate * 1e6  # → 100 μs
range_m = delay_time_us * 3e8 / 2e6  # ≈ 15 km

# SNR to noise power ratio
snr_db = 10
noise_power_ratio = 10 ** -(snr_db / 10)  # → 0.1
```

---

## Example Execution

```bash
# Run all examples
python examples_adversarial_attacks.py

# Example 1: Individual attacks
# ...demonstrates each attack type

# Example 2: Full attack suite
# ...trains model, runs suite, shows results

# Example 3: Custom analysis
# ...robustness study across SNR levels
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `src/adversarial_attacks.py` | Main module (588 lines) |
| `ADVERSARIAL_ATTACKS.md` | API reference & theory |
| `ADVERSARIAL_INTEGRATION_GUIDE.md` | Integration patterns |
| `examples_adversarial_attacks.py` | Full example suite |

---

## Integration Checklist

- [ ] Import: `from src.adversarial_attacks import ...`
- [ ] Create: `attacks = AdversarialRadarAttacks()`
- [ ] Test: `attacked = attacks.add_gaussian_noise(signal, 10)`
- [ ] Eval: `results = run_attack_suite(model, signals, labels)`
- [ ] Save: `json.dump(results, open('results.json', 'w'))`
- [ ] Visualize: Plot results for presentation

---

## Key Papers & References

- Adversarial Examples: [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)
- Radar Spoofing: [Shepard et al., 2012](https://doi.org/10.21236/ADA603272)
- Radar Signal Processing: [Skolnik, 2008]

---

**Last Updated:** 2026-02-20  
**Version:** 1.0  
**Status:** ✅ Production-Ready

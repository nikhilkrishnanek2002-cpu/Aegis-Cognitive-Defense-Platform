# Adversarial Radar Attack Simulations Module

## Overview

The `adversarial_attacks.py` module implements realistic adversarial attack simulations for evaluating radar AI model robustness. Designed for research experiments, model hardening, and vulnerability analysis.

**Location:** `src/adversarial_attacks.py`

## Features

✅ **Five Attack Types**
- Gaussian noise (AWGN) at specified SNR
- Frequency shift (Doppler/heterodyne effects)
- Amplitude scaling (power manipulation)
- Replay attacks (multipath/echo)
- Spoofing attacks (fake target generation)

✅ **Research-Grade Design**
- Physical signal processing (I/Q signals)
- Realistic radar parameters (carrier frequency, sample rate)
- Configurable attack parameters
- Per-attack and summary statistics

✅ **Comprehensive Suite Evaluation**
- Baseline accuracy on clean signals
- Per-attack accuracy degradation
- Confusion matrices for each attack
- Robustness metrics and statistical summaries

## Installation

No additional dependencies beyond project requirements:
- `numpy` (arrays and signal processing)
- `torch` (model inference)
- `scikit-learn` (metrics)

## API Reference

### Core Class: `AdversarialRadarAttacks`

```python
from src.adversarial_attacks import AdversarialRadarAttacks

attacks = AdversarialRadarAttacks(logger=None)
```

### 1. `add_gaussian_noise(signal, snr_db)`

Add Additive White Gaussian Noise (AWGN) at specified SNR.

**Parameters:**
- `signal` (np.ndarray): Input I/Q signal, shape (n_samples,) or (n_samples, 2)
- `snr_db` (float): Signal-to-Noise Ratio in dB

**Returns:**
- `np.ndarray`: Noisy signal with same shape as input

**Physical Model:**
```
SNR (dB) = 10 * log₁₀(P_signal / P_noise)
P_noise = P_signal / 10^(SNR_dB/10)
Noise = I + j*Q distributed as N(0, σ²)
```

**Example:**
```python
from src.adversarial_attacks import add_gaussian_noise

# Add 10 dB AWGN
noisy_signal = add_gaussian_noise(signal, snr_db=10)
```

**Use Cases:**
- Model robustness to channel noise
- Environmental degradation simulation
- Natural background jamming

---

### 2. `frequency_shift_attack(signal, shift_hz, sample_rate=1e6)`

Apply frequency shift (Doppler effect or heterodyne attack).

**Parameters:**
- `signal` (np.ndarray): Input I/Q signal
- `shift_hz` (float): Frequency shift in Hz (positive = upshift)
- `sample_rate` (float): Sampling rate in Hz

**Returns:**
- `np.ndarray`: Frequency-shifted I/Q signal

**Physical Model:**
```
y(t) = x(t) * exp(j * 2π * f_shift * t)
Doppler_freq = 2 * f_c * v / c    (v = velocity, c = speed of light)
```

**Example:**
```python
# Simulate 500 Hz Doppler shift (e.g., 22 m/s target at 10 GHz)
shifted = frequency_shift_attack(signal, shift_hz=500)

# Doppler from velocity
doppler_freq = 2 * 10e9 * 22 / 3e8  # ≈ 1.47 kHz
shifted = frequency_shift_attack(signal, shift_hz=doppler_freq)
```

**Use Cases:**
- Doppler shift confusion
- Frequency translation attacks
- Velocity estimation evasion

**Related Physics:**
- Radar Doppler: f_d = 2 * f_c * v / c
- For 10 GHz radar, 1 m/s velocity ≈ 67 Hz shift

---

### 3. `amplitude_scaling_attack(signal, scale)`

Scale amplitude of signal (power manipulation attack).

**Parameters:**
- `signal` (np.ndarray): Input I/Q signal
- `scale` (float): Multiplicative scaling factor

**Returns:**
- `np.ndarray`: Amplitude-scaled signal

**Physical Model:**
```
y(t) = scale * x(t)
Power_out = scale² * Power_in
```

**Example:**
```python
# Attenuate signal to 50% power
attenuated = amplitude_scaling_attack(signal, scale=0.5)

# Amplify signal by 50%
amplified = amplitude_scaling_attack(signal, scale=1.5)
```

**Use Cases:**
- Power fade/multipath loss
- Gain control exploitation
- Range ambiguity injection

---

### 4. `replay_attack(signal, delay_samples)`

Replay delayed copy of signal (echo/multipath attack).

**Parameters:**
- `signal` (np.ndarray): Input I/Q signal
- `delay_samples` (int): Number of samples to delay

**Returns:**
- `np.ndarray`: Signal with replayed delayed copy

**Physical Model:**
```
y(t) = x(t) + α * x(t - τ)    (α ≈ 0.6 for path loss)
Range_delay = delay_samples / sample_rate * c / 2
```

**Example:**
```python
# Add 50-sample delayed echo (5 microseconds at 1 MHz)
echoed = replay_attack(signal, delay_samples=50)

# Range of echo: 5 μs * 3e8 m/s / 2 ≈ 750 m
```

**Use Cases:**
- Ghost target generation
- Multipath propagation
- Spoofing via coherent injection

---

### 5. `spoof_target_attack(signal, fake_range, fake_velocity, carrier_freq=10e9, sample_rate=1e6)`

Generate spoofed target via range and velocity encoding.

**Parameters:**
- `signal` (np.ndarray): Input I/Q signal (reference for amplitude)
- `fake_range` (float): Fake target range in meters
- `fake_velocity` (float): Fake target velocity in m/s
- `carrier_freq` (float): Radar carrier frequency in Hz
- `sample_rate` (float): Sampling rate in Hz

**Returns:**
- `np.ndarray`: Signal with encoded fake target

**Physical Model:**
```
Range phase = 4π * f_c * R / c
Doppler phase = 2π * f_c * v * t / c
Spoofed = ref_power * exp(j * (phase_range + phase_doppler))
```

**Example:**
```python
# Spoof target at 500m range, 50 m/s velocity
spoofed = spoof_target_attack(
    signal,
    fake_range=500,
    fake_velocity=50,
    carrier_freq=10e9,
    sample_rate=1e6
)
```

**Spoofing Calculations:**
```
For 10 GHz radar:
  500m range ≈ 3,333 ns round-trip delay
  50 m/s ≈ 3.33 kHz Doppler shift
```

**Use Cases:**
- Receiver spoofing simulation
- Coherent false target injection
- GPS/navigation denial research

---

### 6. `run_attack_suite(model, signals, labels, attack_params=None, device='cpu', batch_size=32)`

Execute comprehensive adversarial attack evaluation suite.

**Parameters:**
- `model` (torch.nn.Module): Trained neural network model
- `signals` (np.ndarray): Input signals, shape (n_samples, n_features)
- `labels` (np.ndarray): Ground truth labels
- `attack_params` (dict): Attack configuration (see below)
- `device` (str): 'cpu' or 'cuda'
- `batch_size` (int): Inference batch size

**Returns:**
- `dict`: Comprehensive results with structure:
  ```python
  {
      'baseline_accuracy': 0.95,
      'baseline_conf_matrix': [[...], [...], ...],
      'attacks': {
          'gaussian_noise': {
              'snr_5_db': {'accuracy': 0.92, ...},
              'snr_10_db': {'accuracy': 0.94, ...},
              'mean_accuracy': 0.93,
              'min_accuracy': 0.92,
              ...
          },
          'frequency_shift': {...},
          ...
      },
      'summary': {
          'baseline_accuracy': 0.95,
          'mean_attacked_accuracy': 0.87,
          'accuracy_drop_mean': 0.08,
          'robustness_score': 0.916,
          ...
      },
      'execution_time': 125.43
  }
  ```

**Default Attack Parameters:**
```python
{
    'noise_snr_db': [5, 10, 15, 20],           # AWGN levels
    'freq_shift_hz': [100, 500],               # Frequency shifts
    'amplitude_scales': [0.5, 0.8, 1.2],       # Power scaling
    'replay_delays_samples': [100, 500],       # Echo delays
    'spoof_ranges': [100, 500],                # Fake ranges (m)
    'spoof_velocities': [10, 50],              # Fake velocities (m/s)
}
```

**Example:**
```python
from src.adversarial_attacks import run_attack_suite
import torch

# Run full suite
results = run_attack_suite(
    model=model,
    signals=test_signals,
    labels=test_labels,
    attack_params={
        'noise_snr_db': [5, 10, 15],
        'freq_shift_hz': [200, 400],
    },
    device='cuda',
    batch_size=64
)

# Access results
print(f"Baseline: {results['baseline_accuracy']:.3f}")
print(f"Noise robustness: {results['attacks']['gaussian_noise']['mean_accuracy']:.3f}")
print(f"Overall robustness: {results['summary']['robustness_score']:.3f}")
```

---

## Output Structure

### Attack Suite Results Dictionary

```
results['baseline_accuracy']
  → Clean signal classification accuracy

results['baseline_conf_matrix']
  → Confusion matrix for clean signals

results['attacks'][attack_name]
  → Results for specific attack
  → Contains per-parameter-set results
  → May include mean/min/max statistics

results['summary']
  → Cross-attack statistics:
    - baseline_accuracy
    - mean_attacked_accuracy
    - min_attacked_accuracy
    - accuracy_drop_mean
    - robustness_score (attacked_acc / baseline_acc)
    - total_accuracy_variance

results['execution_time']
  → Total computation time in seconds
```

## Research Applications

### 1. Model Robustness Assessment
```python
results = run_attack_suite(model, signals, labels)
robustness = results['summary']['robustness_score']
if robustness > 0.9:
    print("✅ Model is robust to adversarial attacks")
else:
    print("⚠️  Model needs hardening")
```

### 2. Vulnerability Analysis
```python
# Find most damaging attack
worst_attack = min(
    results['attacks'].items(),
    key=lambda x: x[1]['mean_accuracy']
)
print(f"Most damaging: {worst_attack[0]}")
```

### 3. Parameter Sensitivity Study
```python
# Vary noise levels
results = run_attack_suite(
    model, signals, labels,
    attack_params={
        'noise_snr_db': list(range(0, 25, 5))
    }
)
# Plot accuracy vs SNR
import matplotlib.pyplot as plt
snrs = [int(k.split('_')[1]) for k in results['attacks']['gaussian_noise'].keys()]
accs = [v['accuracy'] for v in results['attacks']['gaussian_noise'].values()]
plt.plot(snrs, accs)
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')
plt.show()
```

### 4. Defense Mechanism Testing
```python
# Test original model
results_original = run_attack_suite(model_original, signals, labels)

# Test hardened model
results_hardened = run_attack_suite(model_hardened, signals, labels)

# Compare
print(f"Original robustness: {results_original['summary']['robustness_score']:.3f}")
print(f"Hardened robustness: {results_hardened['summary']['robustness_score']:.3f}")
```

## Testing

Run the example script:
```bash
python examples_adversarial_attacks.py
```

This generates:
- Example 1: Individual attack demonstrations
- Example 2: Full attack suite evaluation with model training
- Example 3: Custom attack analysis workflows
- JSON results: `results/adversarial_attack_results.json`

## Performance Notes

| Metric | Value |
|--------|-------|
| Attack function execution | <1 ms per signal |
| Full suite (50 signals) | ~100 ms |
| Model inference | ~1 ms per batch (CPU) |
| Memory per signal | ~8 KB (1000-sample I/Q) |

## Common Use Patterns

### Lightweight Testing
```python
# Single attack type
attacked = add_gaussian_noise(signal, 10)
result = model(torch.FloatTensor(attacked))
```

### Research Experiments
```python
# Full suite with custom parameters
results = run_attack_suite(
    model, signals, labels,
    attack_params={'noise_snr_db': np.arange(0, 25, 2)},
    device='cuda', batch_size=128
)
json.dump(results, open('experiment_results.json', 'w'))
```

### Integration with Training
```python
# Adversarial training loop
for epoch in range(epochs):
    for batch_signals, batch_labels in dataloader:
        # Apply random attack
        attack_type = np.random.choice(['noise', 'freq_shift', 'amplitude'])
        if attack_type == 'noise':
            batch_signals = add_gaussian_noise(batch_signals, snr_db=10)
        # ... train on attacked signals
```

## Physical Interpretations

| Attack | Physical Meaning | Radar Effect |
|--------|-----------------|--------------|
| Gaussian Noise | AWGN channel | Signal degradation |
| Frequency Shift | Doppler effect | Velocity estimation error |
| Amplitude Scaling | Gain/path loss | Range ambiguity |
| Replay | Multipath/echo | Ghost target |
| Spoofing | Coherent injection | False detection |

## Citation

If using this module in published research:
```
@misc{RadarAIAttacks,
  title={Adversarial Radar Signal Attack Simulations},
  author={Aegis Cognitive Defense Platform},
  year={2026}
}
```

## References

- Radar Signal Processing: Skolnik, M. I. (2008). Radar Handbook
- Adversarial ML: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
- Spoofing: Shepard et al., "Evaluation of the Vulnerability of GPS to Signal Spoofing" (ION 2012)

---

**Module Version:** 1.0
**Last Updated:** 2026-02-20
**Compatible with:** Python 3.8+, PyTorch 1.9+

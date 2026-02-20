"""
Adversarial Attack Examples and Demonstrations

Shows how to use each attack function and run the full attack suite
with synthetic models and data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
import torch.nn as nn
from adversarial_attacks import AdversarialRadarAttacks, run_attack_suite
import json
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRadarModel(nn.Module):
    """Simple CNN model for radar signal classification (3 classes)."""
    
    def __init__(self, input_size=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * (input_size // 4), 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x shape: (batch, seq_len) or (batch, 2, seq_len) for I/Q
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def generate_synthetic_radar_signals(n_signals=100, n_samples=256):
    """Generate synthetic radar I/Q signals for testing."""
    signals = []
    labels = []
    
    for class_id in range(3):
        for _ in range(n_signals // 3):
            # Generate IQ signal with class-specific characteristics
            t = np.arange(n_samples) / 1e6  # 1 MHz sample rate
            
            if class_id == 0:
                # Class 0: Person walking (low frequency)
                signal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.cos(2 * np.pi * 200 * t)
            elif class_id == 1:
                # Class 1: Vehicle (mid frequency)
                signal = np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
            else:
                # Class 2: Drone (high frequency)
                signal = np.sin(2 * np.pi * 1500 * t) + 0.4 * np.cos(2 * np.pi * 3000 * t)
            
            # Add some noise
            signal += 0.1 * np.random.randn(n_samples)
            
            signals.append(signal)
            labels.append(class_id)
    
    return np.array(signals), np.array(labels)


def example_individual_attacks():
    """
    Example 1: Individual attack functions
    
    Demonstrates each attack type on a single signal.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: INDIVIDUAL ATTACK FUNCTIONS")
    print("="*70)
    
    # Generate test signal
    signal = np.sin(2 * np.pi * np.linspace(0, 10, 256))
    print(f"\nOriginal signal shape: {signal.shape}")
    print(f"Original signal power: {np.mean(signal**2):.4f}")
    
    attacks = AdversarialRadarAttacks(logger=logger)
    
    # 1. Gaussian Noise
    print("\n1️⃣  GAUSSIAN NOISE ATTACK")
    noisy = attacks.add_gaussian_noise(signal, snr_db=10)
    print(f"   SNR: 10 dB")
    print(f"   Noisy signal power: {np.mean(noisy**2):.4f}")
    print(f"   Signal distortion: {np.mean((signal - noisy)**2):.4f}")
    
    # 2. Frequency Shift
    print("\n2️⃣  FREQUENCY SHIFT ATTACK")
    freq_shifted = attacks.frequency_shift_attack(signal, shift_hz=500, sample_rate=1e6)
    print(f"   Frequency shift: 500 Hz")
    print(f"   Output power: {np.mean(np.abs(freq_shifted)**2):.4f}")
    
    # 3. Amplitude Scaling
    print("\n3️⃣  AMPLITUDE SCALING ATTACK")
    for scale in [0.5, 0.8, 1.2]:
        scaled = attacks.amplitude_scaling_attack(signal, scale=scale)
        print(f"   Scale {scale}: Power = {np.mean(scaled**2):.4f}")
    
    # 4. Replay Attack
    print("\n4️⃣  REPLAY ATTACK")
    replayed = attacks.replay_attack(signal, delay_samples=50)
    print(f"   Delay: 50 samples")
    print(f"   Output power: {np.mean(replayed**2):.4f}")
    
    # 5. Spoofing Attack
    print("\n5️⃣  SPOOFING ATTACK")
    spoofed = attacks.spoof_target_attack(
        signal, fake_range=100, fake_velocity=20
    )
    print(f"   Fake range: 100 m")
    print(f"   Fake velocity: 20 m/s")
    print(f"   Output power: {np.mean(np.abs(spoofed)**2):.4f}")
    
    print("\n✅ Individual attacks completed\n")


def example_attack_suite_evaluation():
    """
    Example 2: Full attack suite evaluation
    
    Runs all attacks against a trained model and collects statistics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: FULL ATTACK SUITE EVALUATION")
    print("="*70)
    
    # Generate synthetic data
    print("\nGenerating synthetic radar signals...")
    signals, labels = generate_synthetic_radar_signals(n_signals=50, n_samples=256)
    print(f"   Generated {signals.shape[0]} signals of shape {signals.shape[1:]}")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # Create and train model
    print("\nInitializing SimpleRadarModel...")
    model = SimpleRadarModel(input_size=256)
    
    # Simple training loop (just 1 epoch for demo)
    print("Training model (1 epoch for demo)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(1):
        for batch_signals, batch_labels in _batch_data(signals, labels, batch_size=16):
            batch_tensor = torch.FloatTensor(batch_signals).unsqueeze(1)
            labels_tensor = torch.LongTensor(batch_labels)
            
            optimizer.zero_grad()
            outputs = model(batch_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
    
    model.eval()
    print(f"   Training complete. Model ready.")
    
    # Define attack parameters
    attack_params = {
        'noise_snr_db': [5, 10, 15],
        'freq_shift_hz': [200, 500],
        'amplitude_scales': [0.5, 1.0, 1.5],
        'replay_delays_samples': [50, 100],
        'spoof_ranges': [100, 500],
        'spoof_velocities': [10, 30],
    }
    
    # Run attack suite
    print("\n🎯 Running Attack Suite...")
    results = run_attack_suite(
        model=model,
        signals=signals,
        labels=labels,
        attack_params=attack_params,
        device='cpu',
        batch_size=16
    )
    
    # Display results
    _print_results(results)
    
    return results


def example_custom_attack_analysis():
    """
    Example 3: Custom attack analysis workflow
    
    Show how to use attack suite for research experiment.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: CUSTOM ATTACK ANALYSIS WORKFLOW")
    print("="*70)
    
    # Setup
    signals, labels = generate_synthetic_radar_signals(n_signals=30, n_samples=256)
    model = SimpleRadarModel(input_size=256)
    
    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1):
        for batch_signals, batch_labels in _batch_data(signals, labels, batch_size=16):
            batch_tensor = torch.FloatTensor(batch_signals).unsqueeze(1)
            labels_tensor = torch.LongTensor(batch_labels)
            optimizer.zero_grad()
            outputs = model(batch_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
    
    model.eval()
    
    # Scenario 1: Noise robustness analysis
    print("\n📊 Scenario 1: Noise Robustness Analysis")
    print("-" * 50)
    noise_levels = np.arange(0, 25, 5)
    accuracies = []
    
    attacks = AdversarialRadarAttacks(logger=logger)
    for snr in noise_levels:
        attacked_signals = np.array([
            attacks.add_gaussian_noise(sig, snr) for sig in signals
        ])
        
        with torch.no_grad():
            outputs = model(torch.FloatTensor(attacked_signals).unsqueeze(1))
            preds = outputs.argmax(dim=1).numpy()
        
        acc = (preds == labels).mean()
        accuracies.append(acc)
        print(f"   SNR {snr:2d} dB: Accuracy = {acc:.3f}")
    
    # Scenario 2: Attack severity comparison
    print("\n📊 Scenario 2: Attack Severity Comparison")
    print("-" * 50)
    
    attack_types = {
        'Gaussian Noise (SNR=10)': lambda s: attacks.add_gaussian_noise(s, 10),
        'Freq Shift (500 Hz)': lambda s: attacks.frequency_shift_attack(s, 500),
        'Amplitude Scale (0.5)': lambda s: attacks.amplitude_scaling_attack(s, 0.5),
        'Replay (delay=100)': lambda s: attacks.replay_attack(s, 100),
    }
    
    baseline_acc = (np.argmax(model(torch.FloatTensor(signals).unsqueeze(1)).detach().numpy(), axis=1) == labels).mean()
    
    for attack_name, attack_func in attack_types.items():
        attacked_signals = np.array([attack_func(sig) for sig in signals])
        
        with torch.no_grad():
            outputs = model(torch.FloatTensor(attacked_signals).unsqueeze(1))
            preds = outputs.argmax(dim=1).numpy()
        
        acc = (preds == labels).mean()
        drop = (baseline_acc - acc) * 100
        print(f"   {attack_name:30s}: {acc:.3f} (↓ {drop:.1f}%)")
    
    print(f"\n   Baseline (clean): {baseline_acc:.3f}")


def _batch_data(signals, labels, batch_size):
    """Helper to create batches."""
    for i in range(0, len(signals), batch_size):
        yield signals[i:i+batch_size], labels[i:i+batch_size]


def _print_results(results):
    """Pretty print attack suite results."""
    print("\n" + "="*70)
    print("ATTACK SUITE RESULTS")
    print("="*70)
    
    print(f"\n📌 BASELINE PERFORMANCE")
    print(f"   Accuracy (clean signals): {results['baseline_accuracy']:.3f}")
    
    print(f"\n📌 ATTACK RESULTS")
    for attack_name, attack_results in results['attacks'].items():
        print(f"\n   {attack_name.upper()}")
        # Print statistics if available
        stats_to_print = ['mean_accuracy', 'min_accuracy', 'max_accuracy']
        for stat in stats_to_print:
            if stat in attack_results:
                val = attack_results[stat]
                stat_name = stat.replace('_', ' ').title()
                print(f"      {stat_name}: {val:.3f}")
        
        # If no stats, show individual results
        if not any(s in attack_results for s in stats_to_print):
            count = 0
            for key, value in attack_results.items():
                if isinstance(value, dict) and 'accuracy' in value and count < 3:
                    print(f"      {key}: {value['accuracy']:.3f}")
                    count += 1
    
    print(f"\n📌 SUMMARY STATISTICS")
    for key, value in results['summary'].items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    print(f"\n⏱️  Execution time: {results['execution_time']:.2f} seconds")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ADVERSARIAL RADAR ATTACK EXAMPLES")
    print("="*70)
    
    # Run all examples
    example_individual_attacks()
    results = example_attack_suite_evaluation()
    example_custom_attack_analysis()
    
    # Save results
    output_file = "results/adversarial_attack_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_file}")
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")

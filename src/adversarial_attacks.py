"""
Adversarial Radar Signal Attack Simulations

Implements realistic adversarial attack functions for evaluating radar AI
model robustness against common jamming, spoofing, and signal degradation
attacks. Designed for research experiments and model hardening studies.

Functions:
    - add_gaussian_noise: AWGN at specified SNR
    - frequency_shift_attack: Doppler/frequency manipulation
    - amplitude_scaling_attack: Power level manipulation
    - replay_attack: Time-delayed signal repetition
    - spoof_target_attack: Fake target generation via range/velocity encoding
    - run_attack_suite: Full suite evaluation framework
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import json
import time


class AdversarialRadarAttacks:
    """
    Comprehensive suite of adversarial attacks for radar signal evaluation.
    
    Implements research-grade signal processing attacks suitable for
    studying model robustness and vulnerability analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize attack suite.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for progress tracking
        """
        self.logger = logger or self._setup_default_logger()
        self.attack_metrics = {}
    
    @staticmethod
    def _setup_default_logger():
        """Setup basic logger if none provided."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    # =====================================================================
    # INDIVIDUAL ATTACK FUNCTIONS
    # =====================================================================
    
    def add_gaussian_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add Additive White Gaussian Noise (AWGN) at specified SNR.
        
        Simulates natural channel noise and jamming conditions. Commonly used
        as baseline degradation attack for radar signal analysis.
        
        Parameters
        ----------
        signal : np.ndarray
            Input I/Q signal, shape (n_samples,) or (n_samples, 2)
        snr_db : float
            Signal-to-Noise Ratio in dB
        
        Returns
        -------
        np.ndarray
            Noisy signal with same shape as input
        
        Notes
        -----
        SNR conversion: P_noise = P_signal / 10^(SNR_dB/10)
        """
        signal = np.atleast_1d(signal)
        
        # Handle complex or real signals
        if signal.ndim == 2:
            # I/Q format: (n_samples, 2)
            signal_complex = signal[:, 0] + 1j * signal[:, 1]
        else:
            signal_complex = signal
        
        # Calculate signal power
        signal_power = np.mean(np.abs(signal_complex) ** 2)
        
        if signal_power == 0:
            self.logger.warning("Input signal has zero power, returning as-is")
            return signal
        
        # Calculate noise power from SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate AWGN
        noise_i = np.random.normal(0, np.sqrt(noise_power / 2), len(signal_complex))
        noise_q = np.random.normal(0, np.sqrt(noise_power / 2), len(signal_complex))
        noise = noise_i + 1j * noise_q
        
        # Add noise
        noisy_signal = signal_complex + noise
        
        # Convert back to original format
        if signal.ndim == 2:
            return np.column_stack([np.real(noisy_signal), np.imag(noisy_signal)])
        else:
            return noisy_signal
    
    def frequency_shift_attack(self, signal: np.ndarray, shift_hz: float,
                               sample_rate: float = 1e6) -> np.ndarray:
        """
        Apply frequency shift (Doppler effect or heterodyne attack).
        
        Simulates either natural Doppler shift or active frequency translation
        attack to mislead velocity/frequency estimation in radar systems.
        
        Parameters
        ----------
        signal : np.ndarray
            Input I/Q signal
        shift_hz : float
            Frequency shift in Hz (positive = upshift, negative = downshift)
        sample_rate : float, default=1e6
            Sampling rate in Hz
        
        Returns
        -------
        np.ndarray
            Frequency-shifted I/Q signal
        
        Notes
        -----
        Implements mixing: s(t) * exp(j*2*pi*f_shift*t)
        """
        signal = np.atleast_1d(signal)
        
        # Handle I/Q format
        if signal.ndim == 2:
            signal_complex = signal[:, 0] + 1j * signal[:, 1]
        else:
            signal_complex = signal
        
        # Generate complex exponential for frequency shift
        t = np.arange(len(signal_complex)) / sample_rate
        phase_shift = np.exp(1j * 2 * np.pi * shift_hz * t)
        
        # Apply frequency translation
        shifted_signal = signal_complex * phase_shift
        
        # Convert back to original format
        if signal.ndim == 2:
            return np.column_stack([np.real(shifted_signal), np.imag(shifted_signal)])
        else:
            return shifted_signal
    
    def amplitude_scaling_attack(self, signal: np.ndarray, scale: float) -> np.ndarray:
        """
        Scale amplitude of signal (power manipulation attack).
        
        Simulates attacker control over transmit power or reflection strength.
        Can model power fade, amplification, or jamming power control.
        
        Parameters
        ----------
        signal : np.ndarray
            Input I/Q signal
        scale : float
            Multiplicative scaling factor (1.0 = no attack)
        
        Returns
        -------
        np.ndarray
            Amplitude-scaled I/Q signal
        
        Notes
        -----
        Simple multiplication: y(t) = scale * x(t)
        Values > 1.0 amplify, < 1.0 attenuate
        """
        if scale < 0:
            self.logger.warning(f"Negative scale {scale} provided, using absolute value")
            scale = abs(scale)
        
        return signal * scale
    
    def replay_attack(self, signal: np.ndarray, delay_samples: int) -> np.ndarray:
        """
        Replay delayed copy of signal (echo/multipath attack).
        
        Simulates ghost target from replayed signals or multipath propagation.
        Common in spoofing scenarios with coherent signal injection.
        
        Parameters
        ----------
        signal : np.ndarray
            Input I/Q signal
        delay_samples : int
            Number of samples to delay (delay_time = delay_samples / sample_rate)
        
        Returns
        -------
        np.ndarray
            Signal with replayed delayed copy at reduced amplitude
        
        Notes
        -----
        Implements: y(t) = x(t) + alpha * x(t - delay)
        Alpha typically 0.5-0.8 to model path loss
        """
        signal = np.atleast_1d(signal)
        
        if delay_samples <= 0:
            self.logger.warning("Delay must be positive, returning original signal")
            return signal
        
        if delay_samples >= len(signal):
            self.logger.warning(f"Delay {delay_samples} >= signal length {len(signal)}")
            return signal
        
        # Create output signal
        attacked_signal = signal.copy()
        
        # Add delayed replica with alpha attenuation
        alpha = 0.6  # Path loss factor
        attacked_signal[delay_samples:] += alpha * signal[:-delay_samples]
        
        return attacked_signal
    
    def spoof_target_attack(self, signal: np.ndarray, fake_range: float,
                           fake_velocity: float, carrier_freq: float = 10e9,
                           sample_rate: float = 1e6) -> np.ndarray:
        """
        Generate spoofed target via range and velocity encoding.
        
        Creates synthetic I/Q signal encoding fake target at specified
        range and velocity. Models receiver-based spoofing attack.
        Signal structure: exp(j*(range_phase + doppler_phase))
        
        Parameters
        ----------
        signal : np.ndarray
            Input I/Q signal (used for amplitude/power reference)
        fake_range : float
            Fake target range in meters
        fake_velocity : float
            Fake target velocity in m/s
        carrier_freq : float, default=10e9
            Radar carrier frequency in Hz
        sample_rate : float, default=1e6
            Sampling rate in Hz
        
        Returns
        -------
        np.ndarray
            Spoofed I/Q signal with encoded range/velocity
        
        Notes
        -----
        Range phase: 4*pi*f_c*range / c (c = 3e8)
        Doppler phase: 2*pi*f_c*velocity*t / c
        """
        signal = np.atleast_1d(signal)
        
        # Get signal length and reference power
        n_samples = len(signal)
        if signal.ndim == 2:
            signal_complex = signal[:, 0] + 1j * signal[:, 1]
        else:
            signal_complex = signal
        
        ref_power = np.mean(np.abs(signal_complex) ** 2)
        
        # Speed of light
        c = 3e8
        
        # Range encoding (phase)
        range_phase = 4 * np.pi * carrier_freq * fake_range / c
        
        # Doppler encoding (frequency shift)
        doppler_freq = 2 * carrier_freq * fake_velocity / c
        t = np.arange(n_samples) / sample_rate
        doppler_phase = 2 * np.pi * doppler_freq * t
        
        # Generate spoofed signal
        spoof_signal = np.sqrt(ref_power) * np.exp(1j * (range_phase + doppler_phase))
        
        # Mix with original signal (coherent spoofing)
        attacked_signal = signal_complex + 0.5 * spoof_signal
        
        # Convert back to original format
        if signal.ndim == 2:
            return np.column_stack([np.real(attacked_signal), np.imag(attacked_signal)])
        else:
            return attacked_signal
    
    # =====================================================================
    # ATTACK SUITE ORCHESTRATION
    # =====================================================================
    
    def run_attack_suite(self, model: nn.Module, signals: np.ndarray,
                        labels: np.ndarray,
                        attack_params: Optional[Dict] = None,
                        device: str = 'cpu',
                        batch_size: int = 32) -> Dict:
        """
        Execute comprehensive adversarial attack evaluation suite.
        
        Applies each attack type to test signals and measures model
        robustness via accuracy degradation, per-attack metrics, and
        statistical summaries.
        
        Parameters
        ----------
        model : torch.nn.Module
            Trained neural network model for classification
        signals : np.ndarray
            Array of input signals, shape (n_samples, n_features) or (n_samples, n_features, time)
        labels : np.ndarray
            Ground truth labels, shape (n_samples,)
        attack_params : dict, optional
            Attack-specific parameters. If None, uses defaults:
            - 'noise_snr_db': [5, 10, 15, 20] dB
            - 'freq_shift_hz': [100, 500] Hz
            - 'amplitude_scales': [0.5, 0.8, 1.2]
            - 'replay_delays_samples': [100, 500]
            - 'spoof_ranges': [100, 500] m
            - 'spoof_velocities': [10, 50] m/s
        device : str, default='cpu'
            Device for model inference ('cpu' or 'cuda')
        batch_size : int, default=32
            Batch size for inference
        
        Returns
        -------
        dict
            Comprehensive results dictionary with keys:
            - 'baseline_accuracy': Clean signal accuracy
            - 'baseline_conf_matrix': Clean signal confusion matrix
            - 'attacks': Dict of attack results
            - 'summary': Statistical summary across attacks
            - 'per_attack_metrics': Detailed metrics for each attack
            - 'execution_time': Total execution time in seconds
        
        Examples
        --------
        >>> results = atk.run_attack_suite(
        ...     model=model,
        ...     signals=test_signals,
        ...     labels=test_labels,
        ...     attack_params={
        ...         'noise_snr_db': [10, 15],
        ...         'freq_shift_hz': [200]
        ...     }
        ... )
        >>> print(f"Baseline: {results['baseline_accuracy']:.3f}")
        >>> print(f"Noise attack: {results['attacks']['gaussian_noise']}")
        """
        start_time = time.time()
        
        # Set defaults
        if attack_params is None:
            attack_params = {
                'noise_snr_db': [5, 10, 15, 20],
                'freq_shift_hz': [100, 500],
                'amplitude_scales': [0.5, 0.8, 1.2],
                'replay_delays_samples': [100, 500],
                'spoof_ranges': [100, 500],
                'spoof_velocities': [10, 50],
            }
        
        model.eval()
        model.to(device)
        
        # Get baseline accuracy on clean signals
        self.logger.info("Computing baseline accuracy on clean signals...")
        baseline_acc, baseline_preds, baseline_cm = self._evaluate_model(
            model, signals, labels, device, batch_size
        )
        
        results = {
            'baseline_accuracy': baseline_acc,
            'baseline_conf_matrix': baseline_cm.tolist(),
            'attacks': {},
            'per_attack_metrics': {},
            'execution_time': None,
        }
        
        # Gaussian Noise Attack
        if 'noise_snr_db' in attack_params:
            self.logger.info("Running Gaussian Noise attack...")
            noise_results = self._test_noise_attack(
                model, signals, labels, attack_params['noise_snr_db'],
                device, batch_size
            )
            results['attacks']['gaussian_noise'] = noise_results
        
        # Frequency Shift Attack
        if 'freq_shift_hz' in attack_params:
            self.logger.info("Running Frequency Shift attack...")
            freq_results = self._test_frequency_attack(
                model, signals, labels, attack_params['freq_shift_hz'],
                device, batch_size
            )
            results['attacks']['frequency_shift'] = freq_results
        
        # Amplitude Scaling Attack
        if 'amplitude_scales' in attack_params:
            self.logger.info("Running Amplitude Scaling attack...")
            amp_results = self._test_amplitude_attack(
                model, signals, labels, attack_params['amplitude_scales'],
                device, batch_size
            )
            results['attacks']['amplitude_scaling'] = amp_results
        
        # Replay Attack
        if 'replay_delays_samples' in attack_params:
            self.logger.info("Running Replay attack...")
            replay_results = self._test_replay_attack(
                model, signals, labels, attack_params['replay_delays_samples'],
                device, batch_size
            )
            results['attacks']['replay'] = replay_results
        
        # Spoofing Attack
        if 'spoof_ranges' in attack_params or 'spoof_velocities' in attack_params:
            self.logger.info("Running Spoof attack...")
            spoof_results = self._test_spoof_attack(
                model, signals, labels,
                attack_params.get('spoof_ranges', [100, 500]),
                attack_params.get('spoof_velocities', [10, 50]),
                device, batch_size
            )
            results['attacks']['spoof'] = spoof_results
        
        # Compute summary statistics
        results['summary'] = self._compute_summary_statistics(
            results, baseline_acc
        )
        
        results['execution_time'] = time.time() - start_time
        
        self.logger.info(f"Attack suite completed in {results['execution_time']:.2f}s")
        
        return results
    
    # =====================================================================
    # HELPER METHODS FOR ATTACK SUITE
    # =====================================================================
    
    def _evaluate_model(self, model: nn.Module, signals: np.ndarray,
                       labels: np.ndarray, device: str,
                       batch_size: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model and return accuracy, predictions, and confusion matrix."""
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(signals), batch_size):
                batch_signals = signals[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_signals).to(device)
                
                outputs = model(batch_tensor)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)
        
        predictions = np.array(predictions)
        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        
        return accuracy, predictions, cm
    
    def _test_noise_attack(self, model: nn.Module, signals: np.ndarray,
                          labels: np.ndarray, snr_values: List[float],
                          device: str, batch_size: int) -> Dict:
        """Test Gaussian noise attack at multiple SNR levels."""
        results = {}
        accuracies = []
        
        for snr_db in snr_values:
            attacked_signals = np.array([
                self.add_gaussian_noise(sig, snr_db) for sig in signals
            ])
            
            acc, preds, cm = self._evaluate_model(
                model, attacked_signals, labels, device, batch_size
            )
            
            results[f'snr_{snr_db}_db'] = {
                'accuracy': float(acc),
                'accuracy_drop': float(1.0 - acc / (np.mean([acc for acc in accuracies]) or 1.0)) if accuracies else 0.0,
                'confusion_matrix': cm.tolist(),
            }
            accuracies.append(acc)
        
        results['mean_accuracy'] = float(np.mean(accuracies))
        results['min_accuracy'] = float(np.min(accuracies))
        results['max_accuracy'] = float(np.max(accuracies))
        
        return results
    
    def _test_frequency_attack(self, model: nn.Module, signals: np.ndarray,
                              labels: np.ndarray, shifts_hz: List[float],
                              device: str, batch_size: int) -> Dict:
        """Test frequency shift attack."""
        results = {}
        accuracies = []
        
        for shift_hz in shifts_hz:
            attacked_signals = np.array([
                self.frequency_shift_attack(sig, shift_hz) for sig in signals
            ])
            
            acc, preds, cm = self._evaluate_model(
                model, attacked_signals, labels, device, batch_size
            )
            
            results[f'shift_{shift_hz}_hz'] = {
                'accuracy': float(acc),
                'confusion_matrix': cm.tolist(),
            }
            accuracies.append(acc)
        
        results['mean_accuracy'] = float(np.mean(accuracies))
        results['min_accuracy'] = float(np.min(accuracies))
        
        return results
    
    def _test_amplitude_attack(self, model: nn.Module, signals: np.ndarray,
                              labels: np.ndarray, scales: List[float],
                              device: str, batch_size: int) -> Dict:
        """Test amplitude scaling attack."""
        results = {}
        accuracies = []
        
        for scale in scales:
            attacked_signals = np.array([
                self.amplitude_scaling_attack(sig, scale) for sig in signals
            ])
            
            acc, preds, cm = self._evaluate_model(
                model, attacked_signals, labels, device, batch_size
            )
            
            results[f'scale_{scale}'] = {
                'accuracy': float(acc),
                'confusion_matrix': cm.tolist(),
            }
            accuracies.append(acc)
        
        results['mean_accuracy'] = float(np.mean(accuracies))
        results['min_accuracy'] = float(np.min(accuracies))
        
        return results
    
    def _test_replay_attack(self, model: nn.Module, signals: np.ndarray,
                           labels: np.ndarray, delays: List[int],
                           device: str, batch_size: int) -> Dict:
        """Test replay attack."""
        results = {}
        accuracies = []
        
        for delay in delays:
            attacked_signals = np.array([
                self.replay_attack(sig, delay) for sig in signals
            ])
            
            acc, preds, cm = self._evaluate_model(
                model, attacked_signals, labels, device, batch_size
            )
            
            results[f'delay_{delay}_samples'] = {
                'accuracy': float(acc),
                'confusion_matrix': cm.tolist(),
            }
            accuracies.append(acc)
        
        results['mean_accuracy'] = float(np.mean(accuracies))
        results['min_accuracy'] = float(np.min(accuracies))
        
        return results
    
    def _test_spoof_attack(self, model: nn.Module, signals: np.ndarray,
                          labels: np.ndarray, ranges: List[float],
                          velocities: List[float], device: str,
                          batch_size: int) -> Dict:
        """Test spoofing attack."""
        results = {}
        accuracies = []
        
        for fake_range in ranges:
            for fake_velocity in velocities:
                attacked_signals = np.array([
                    self.spoof_target_attack(sig, fake_range, fake_velocity)
                    for sig in signals
                ])
                
                acc, preds, cm = self._evaluate_model(
                    model, attacked_signals, labels, device, batch_size
                )
                
                key = f'range_{fake_range}m_vel_{fake_velocity}ms'
                results[key] = {
                    'accuracy': float(acc),
                    'confusion_matrix': cm.tolist(),
                }
                accuracies.append(acc)
        
        results['mean_accuracy'] = float(np.mean(accuracies))
        results['min_accuracy'] = float(np.min(accuracies))
        
        return results
    
    def _compute_summary_statistics(self, results: Dict,
                                   baseline_acc: float) -> Dict:
        """Compute statistical summary across all attacks."""
        all_accuracies = []
        
        for attack_name, attack_results in results['attacks'].items():
            if isinstance(attack_results, dict):
                for key, value in attack_results.items():
                    if isinstance(value, dict) and 'accuracy' in value:
                        all_accuracies.append(value['accuracy'])
        
        if not all_accuracies:
            return {}
        
        all_accuracies = np.array(all_accuracies)
        
        return {
            'baseline_accuracy': float(baseline_acc),
            'mean_attacked_accuracy': float(np.mean(all_accuracies)),
            'min_attacked_accuracy': float(np.min(all_accuracies)),
            'max_attacked_accuracy': float(np.max(all_accuracies)),
            'accuracy_drop_mean': float(baseline_acc - np.mean(all_accuracies)),
            'accuracy_drop_min': float(baseline_acc - np.min(all_accuracies)),
            'total_accuracy_variance': float(np.var(all_accuracies)),
            'robustness_score': float(np.mean(all_accuracies) / baseline_acc) if baseline_acc > 0 else 0.0,
        }


# Convenience functions
def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Standalone function for Gaussian noise attack."""
    attacks = AdversarialRadarAttacks()
    return attacks.add_gaussian_noise(signal, snr_db)


def frequency_shift_attack(signal: np.ndarray, shift_hz: float,
                          sample_rate: float = 1e6) -> np.ndarray:
    """Standalone function for frequency shift attack."""
    attacks = AdversarialRadarAttacks()
    return attacks.frequency_shift_attack(signal, shift_hz, sample_rate)


def amplitude_scaling_attack(signal: np.ndarray, scale: float) -> np.ndarray:
    """Standalone function for amplitude scaling attack."""
    attacks = AdversarialRadarAttacks()
    return attacks.amplitude_scaling_attack(signal, scale)


def replay_attack(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Standalone function for replay attack."""
    attacks = AdversarialRadarAttacks()
    return attacks.replay_attack(signal, delay_samples)


def spoof_target_attack(signal: np.ndarray, fake_range: float,
                       fake_velocity: float, carrier_freq: float = 10e9,
                       sample_rate: float = 1e6) -> np.ndarray:
    """Standalone function for spoofing attack."""
    attacks = AdversarialRadarAttacks()
    return attacks.spoof_target_attack(signal, fake_range, fake_velocity,
                                       carrier_freq, sample_rate)


def run_attack_suite(model: nn.Module, signals: np.ndarray,
                    labels: np.ndarray,
                    attack_params: Optional[Dict] = None,
                    device: str = 'cpu',
                    batch_size: int = 32) -> Dict:
    """Standalone function for running full attack suite."""
    attacks = AdversarialRadarAttacks()
    return attacks.run_attack_suite(model, signals, labels, attack_params,
                                   device, batch_size)


if __name__ == '__main__':
    print("Adversarial Radar Attacks Module Loaded Successfully")
    print("\nAvailable Functions:")
    print("  - add_gaussian_noise(signal, snr_db)")
    print("  - frequency_shift_attack(signal, shift_hz, sample_rate)")
    print("  - amplitude_scaling_attack(signal, scale)")
    print("  - replay_attack(signal, delay_samples)")
    print("  - spoof_target_attack(signal, fake_range, fake_velocity, ...)")
    print("  - run_attack_suite(model, signals, labels, attack_params, device, batch_size)")

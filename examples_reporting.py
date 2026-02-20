"""
Usage Examples for Reporting Module

This script demonstrates how to use each function in the reporting module
with synthetic data.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from reporting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_training_history,
    plot_detection_vs_snr,
    plot_tracking_rmse
)


def example_confusion_matrix():
    """Example: Confusion Matrix for 3-class classification."""
    print("📊 Generating Confusion Matrix...")
    
    # Synthetic predictions for 3 classes
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 10)
    y_pred = np.array([0, 0, 1, 1, 1, 0, 2, 2, 1] * 10)
    
    save_path = "results/reports/confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, save_path)
    print(f"   ✅ Saved to {save_path}")


def example_roc_curve():
    """Example: ROC Curve for binary classification."""
    print("📈 Generating ROC Curve...")
    
    # Synthetic binary classification data
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.binomial(1, 0.5, n_samples)
    y_prob = y_true * np.random.uniform(0.6, 1.0, n_samples) + \
             (1 - y_true) * np.random.uniform(0.0, 0.4, n_samples)
    
    save_path = "results/reports/roc_curve.png"
    plot_roc_curve(y_true, y_prob, save_path)
    print(f"   ✅ Saved to {save_path}")


def example_precision_recall():
    """Example: Precision-Recall curve."""
    print("🎯 Generating Precision-Recall Curve...")
    
    # Synthetic binary classification data
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.binomial(1, 0.4, n_samples)
    y_prob = y_true * np.random.uniform(0.6, 1.0, n_samples) + \
             (1 - y_true) * np.random.uniform(0.0, 0.3, n_samples)
    
    save_path = "results/reports/precision_recall.png"
    plot_precision_recall(y_true, y_prob, save_path)
    print(f"   ✅ Saved to {save_path}")


def example_training_history():
    """Example: Training and validation curves."""
    print("📉 Generating Training History Plot...")
    
    # Synthetic training history
    epochs = 50
    history = {
        'loss': 2.5 - np.linspace(0, 2.3, epochs) + np.random.normal(0, 0.05, epochs),
        'val_loss': 2.5 - np.linspace(0, 2.0, epochs) + np.random.normal(0, 0.08, epochs),
        'accuracy': np.linspace(0.2, 0.92, epochs) + np.random.normal(0, 0.02, epochs),
        'val_accuracy': np.linspace(0.2, 0.88, epochs) + np.random.normal(0, 0.03, epochs),
    }
    # Clip to valid ranges
    history['accuracy'] = np.clip(history['accuracy'], 0, 1)
    history['val_accuracy'] = np.clip(history['val_accuracy'], 0, 1)
    
    save_path = "results/reports/training_history.png"
    plot_training_history(history, save_path)
    print(f"   ✅ Saved to {save_path}")


def example_detection_vs_snr():
    """Example: Detection accuracy vs SNR."""
    print("📡 Generating Detection vs SNR Plot...")
    
    # Synthetic SNR sweep
    snr_values = np.array([-5, -3, 0, 3, 5, 8, 10, 12, 15, 18, 20])
    accuracy = 1 / (1 + np.exp(-(snr_values + 2) / 3)) + np.random.normal(0, 0.02, len(snr_values))
    accuracy = np.clip(accuracy, 0, 1)
    
    save_path = "results/reports/detection_vs_snr.png"
    plot_detection_vs_snr(snr_values, accuracy, save_path)
    print(f"   ✅ Saved to {save_path}")


def example_tracking_rmse():
    """Example: Tracking RMSE over time."""
    print("🎲 Generating Tracking RMSE Plot...")
    
    # Synthetic RMSE over 1000 time steps
    time = np.arange(1000)
    rmse = 5.0 + 2.0 * np.sin(time / 100) + np.random.normal(0, 0.5, len(time))
    rmse = np.clip(rmse, 0, None)
    
    save_path = "results/reports/tracking_rmse.png"
    plot_tracking_rmse(time, rmse, save_path)
    print(f"   ✅ Saved to {save_path}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("REPORTING MODULE - EXAMPLE USAGE")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs("results/reports", exist_ok=True)
    
    # Run all examples
    example_confusion_matrix()
    example_roc_curve()
    example_precision_recall()
    example_training_history()
    example_detection_vs_snr()
    example_tracking_rmse()
    
    print("\n" + "="*60)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated plots:")
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")
    print("  • precision_recall.png")
    print("  • training_history.png")
    print("  • detection_vs_snr.png")
    print("  • tracking_rmse.png")
    print("\nAll files saved to: results/reports/")
    print("="*60 + "\n")

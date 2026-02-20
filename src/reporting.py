"""
Reporting Module for Radar AI Project

Provides publication-quality visualization functions for performance metrics,
training history, and detection analysis. All plots use scientific styling
suitable for IEEE conference papers.

Functions:
    - plot_confusion_matrix: Classification confusion matrix heatmap
    - plot_roc_curve: Receiver Operating Characteristic curve
    - plot_precision_recall: Precision-Recall curve with F1 contours
    - plot_training_history: Training and validation loss/accuracy
    - plot_detection_vs_snr: Detection performance vs Signal-to-Noise Ratio
    - plot_tracking_rmse: Tracking Root Mean Square Error over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
import os


def _setup_scientific_style():
    """Configure matplotlib for scientific publication quality."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot and save confusion matrix as heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    save_path : str
        Path where figure will be saved (.png, .pdf, or .eps)

    Returns
    -------
    None
        Figure is saved to disk
    """
    _setup_scientific_style()

    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(cm, cmap='Blues', aspect='auto', origin='upper')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20, fontsize=11)

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=10, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: Classification Performance', fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format=os.path.splitext(save_path)[1][1:])
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """
    Plot and save ROC curve with AUC metric.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1)
    y_prob : array-like
        Predicted probabilities for positive class
    save_path : str
        Path where figure will be saved (.png, .pdf, or .eps)

    Returns
    -------
    None
        Figure is saved to disk
    """
    _setup_scientific_style()

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='#1f77b4', lw=2.5,
            label=f'ROC (AUC = {roc_auc:.3f})', zorder=3)

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
            label='Random Classifier (AUC = 0.500)', zorder=2)

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic Curve', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format=os.path.splitext(save_path)[1][1:])
    plt.close()


def plot_precision_recall(y_true, y_prob, save_path):
    """
    Plot and save Precision-Recall curve with F1 contours.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1)
    y_prob : array-like
        Predicted probabilities for positive class
    save_path : str
        Path where figure will be saved (.png, .pdf, or .eps)

    Returns
    -------
    None
        Figure is saved to disk
    """
    _setup_scientific_style()

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot F1 score iso-lines
    f1_scores = np.linspace(0.2, 0.9, 8)
    for f1 in f1_scores:
        x = np.linspace(0.01, 0.99, 100)
        y = (f1 * x) / (2 * x - f1)
        y = np.clip(y, 0, 1)
        ax.plot(x, y, color='lightgray', alpha=0.4, linewidth=0.8, zorder=1)
        # Add label to F1 iso-lines
        if f1 >= 0.3:
            idx_label = 30
            ax.text(x[idx_label], y[idx_label], f'F1={f1:.1f}',
                   fontsize=8, alpha=0.5, rotation=25)

    # Plot PR curve
    ax.plot(recall, precision, color='#2ca02c', lw=2.5,
            label=f'Precision-Recall (AUC = {pr_auc:.3f})', zorder=3)

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve with F1 Score Contours', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format=os.path.splitext(save_path)[1][1:])
    plt.close()


def plot_training_history(history, save_path):
    """
    Plot and save training and validation loss/accuracy over epochs.

    Parameters
    ----------
    history : dict
        Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        or 'loss', 'val_loss', 'accuracy', 'val_accuracy'
    save_path : str
        Path where figure will be saved (.png, .pdf, or .eps)

    Returns
    -------
    None
        Figure is saved to disk
    """
    _setup_scientific_style()

    # Handle different history key formats
    train_loss_key = 'train_loss' if 'train_loss' in history else 'loss'
    train_losses = history.get(train_loss_key, [])
    val_losses = history.get('val_loss')
    if not val_losses:
        val_losses = None

    train_acc_key = 'train_acc' if 'train_acc' in history else 'accuracy'
    train_acc = history.get(train_acc_key)
    val_acc = history.get('val_acc') if 'val_acc' in history else history.get('val_accuracy')
    if val_acc == []:
        val_acc = None

    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, 'o-', color='#1f77b4', lw=2.5,
             markersize=4, label='Training Loss', zorder=3)
    if val_losses is not None:
        ax1.plot(epochs, val_losses, 's-', color='#ff7f0e', lw=2.5,
                 markersize=4, label='Validation Loss', zorder=3)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([epochs[0], epochs[-1]])

    # Accuracy plot
    if train_acc is not None:
        ax2.plot(epochs, train_acc, 'o-', color='#2ca02c', lw=2.5,
                 markersize=4, label='Training Accuracy', zorder=3)
    if val_acc is not None:
        ax2.plot(epochs, val_acc, 's-', color='#d62728', lw=2.5,
                 markersize=4, label='Validation Accuracy', zorder=3)
    if train_acc is None and val_acc is None:
        ax2.axis('off')
    else:
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([epochs[0], epochs[-1]])

    fig.suptitle('Model Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format=os.path.splitext(save_path)[1][1:])
    plt.close()


def plot_detection_vs_snr(snr_values, accuracy, save_path):
    """
    Plot and save detection accuracy vs Signal-to-Noise Ratio.

    Parameters
    ----------
    snr_values : array-like
        SNR values in dB
    accuracy : array-like
        Detection accuracy at each SNR level
    save_path : str
        Path where figure will be saved (.png, .pdf, or .eps)

    Returns
    -------
    None
        Figure is saved to disk
    """
    _setup_scientific_style()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Sort by SNR for proper plotting
    sorted_indices = np.argsort(snr_values)
    snr_sorted = np.array(snr_values)[sorted_indices]
    acc_sorted = np.array(accuracy)[sorted_indices]

    # Plot detection performance
    ax.plot(snr_sorted, acc_sorted, 'o-', color='#1f77b4', lw=2.5,
            markersize=7, label='Detection Accuracy', zorder=3)

    # Add horizontal line at 90% accuracy
    ax.axhline(y=0.90, color='gray', linestyle='--', lw=1.5,
              label='90% Threshold', alpha=0.7, zorder=2)

    ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Radar Detection Performance vs. SNR', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format=os.path.splitext(save_path)[1][1:])
    plt.close()


def plot_tracking_rmse(time, rmse, save_path):
    """
    Plot and save tracking Root Mean Square Error over time.

    Parameters
    ----------
    time : array-like
        Time values (seconds or frame numbers)
    rmse : array-like
        RMSE values (range in meters or normalized)
    save_path : str
        Path where figure will be saved (.png, .pdf, or .eps)

    Returns
    -------
    None
        Figure is saved to disk
    """
    _setup_scientific_style()

    fig, ax = plt.subplots(figsize=(9, 6))

    time = np.array(time)
    rmse = np.array(rmse)

    # Plot RMSE
    ax.plot(time, rmse, '-', color='#d62728', lw=2.5, label='Tracking RMSE', zorder=3)

    # Add mean RMSE line
    mean_rmse = np.mean(rmse)
    ax.axhline(y=mean_rmse, color='gray', linestyle='--', lw=1.5,
              label=f'Mean RMSE = {mean_rmse:.4f}', alpha=0.7, zorder=2)

    # Add shaded region
    ax.fill_between(time, rmse, mean_rmse, where=(rmse >= mean_rmse),
                    alpha=0.2, color='red', interpolate=True, zorder=1)
    ax.fill_between(time, rmse, mean_rmse, where=(rmse < mean_rmse),
                    alpha=0.2, color='green', interpolate=True, zorder=1)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Root Mean Square Error (m)', fontsize=12, fontweight='bold')
    ax.set_title('Target Tracking Root Mean Square Error', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format=os.path.splitext(save_path)[1][1:])
    plt.close()


# Example usage and testing
if __name__ == '__main__':
    print("Reporting Module Loaded Successfully")
    print("\nAvailable Functions:")
    print("  - plot_confusion_matrix(y_true, y_pred, save_path)")
    print("  - plot_roc_curve(y_true, y_prob, save_path)")
    print("  - plot_precision_recall(y_true, y_prob, save_path)")
    print("  - plot_training_history(history, save_path)")
    print("  - plot_detection_vs_snr(snr_values, accuracy, save_path)")
    print("  - plot_tracking_rmse(time, rmse, save_path)")

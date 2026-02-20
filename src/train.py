from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
from data_preprocessing import create_dataset
from model import build_model


# =====================================================================
# REPRODUCIBILITY & SETUP FUNCTIONS
# =====================================================================

def set_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducible training.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value for all libraries
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.run_functions_eagerly(False)
    except ImportError:
        pass


def setup_logging(output_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup structured logging to console and file.
    
    Parameters
    ----------
    output_dir : str
        Directory to save log file
    log_level : int
        Logging level (default: INFO)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger('train')
    logger.setLevel(log_level)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def train(epochs: int = 10,
         batch_size: int = 32,
         test_size: float = 0.2,
         output_dir: str = "results",
         seed: int = 42,
         validation_split: float = 0.2) -> Tuple[Any, Any, Any, Dict]:
    """
    Train model with full reproducibility and logging.
    
    Parameters
    ----------
    epochs : int, default=10
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    test_size : float, default=0.2
        Test set fraction
    output_dir : str, default="results"
        Directory to save checkpoints and logs
    seed : int, default=42
        Random seed for reproducibility
    validation_split : float, default=0.2
        Validation split fraction
    
    Returns
    -------
    model : keras.Model
        Trained model
    Xte : np.ndarray
        Test features
    yte : np.ndarray
        Test labels
    history_dict : dict
        Training history dictionary with keys:
        - 'loss': Training loss per epoch
        - 'val_loss': Validation loss per epoch
        - 'accuracy': Training accuracy per epoch
        - 'val_accuracy': Validation accuracy per epoch
        - 'epoch': Epoch numbers
        - 'epochs': Total epochs
        - 'batch_size': Batch size
        - 'seed': Random seed
        - 'timestamp': Training timestamp
    """
    # =====================================================================
    # SETUP PHASE
    # =====================================================================
    
    # Set reproducibility
    set_seeds(seed)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*70)
    logger.info("🚀 STARTING REPRODUCIBLE TRAINING (Keras/TensorFlow)")
    logger.info("="*70)
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Test size: {test_size}")
    logger.info(f"  Validation split: {validation_split}")
    logger.info(f"  Random seed: {seed}")
    
    # =====================================================================
    # DATA LOADING PHASE
    # =====================================================================
    
    logger.info("📊 Creating dataset...")
    X, y = create_dataset()
    logger.info(f"✓ Dataset created: {X.shape[0]} samples")
    
    logger.info("📊 Splitting data...")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    logger.info(f"✓ Train: {Xtr.shape[0]}, Test: {Xte.shape[0]}")
    
    # =====================================================================
    # MODEL & TRAINING PHASE
    # =====================================================================
    
    logger.info("🧠 Building model...")
    model = build_model()
    logger.info("✓ Model built")
    
    logger.info("📈 Starting training...")
    logger.info("-"*70)
    
    history = model.fit(
        Xtr, ytr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0
    )
    
    # =====================================================================
    # SAVE PHASE
    # =====================================================================
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "radar_model.h5")
    model.save(model_path)
    logger.info(f"✓ Model saved to {model_path}")
    
    # Convert history to dictionary format
    history_dict = {
        'epoch': list(range(1, epochs + 1)),
        'loss': [float(v) for v in history.history.get('loss', [])],
        'val_loss': [float(v) for v in history.history.get('val_loss', [])],
        'accuracy': [float(v) for v in history.history.get('accuracy', [])],
        'val_accuracy': [float(v) for v in history.history.get('val_accuracy', [])],
        'epochs': epochs,
        'batch_size': batch_size,
        'test_size': test_size,
        'validation_split': validation_split,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save history to JSON
    history_file = os.path.join(output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"✓ Training history saved to {history_file}")
    
    # =====================================================================
    # VISUALIZATION PHASE
    # =====================================================================
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['epoch'], history_dict['accuracy'], 'b-o', label="Train", linewidth=2, markersize=5)
    plt.plot(history_dict['epoch'], history_dict['val_accuracy'], 'r-s', label="Val", linewidth=2, markersize=5)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['epoch'], history_dict['loss'], 'b-o', label="Train", linewidth=2, markersize=5)
    plt.plot(history_dict['epoch'], history_dict['val_loss'], 'r-s', label="Val", linewidth=2, markersize=5)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Loss', fontsize=11, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Training plot saved to {plot_path}")
    plt.close()
    
    # =====================================================================
    # SUMMARY PHASE
    # =====================================================================
    
    final_acc = history_dict['accuracy'][-1] if history_dict['accuracy'] else 0
    final_val_acc = history_dict['val_accuracy'][-1] if history_dict['val_accuracy'] else 0
    
    logger.info("📊 Training Summary:")
    logger.info(f"  Final training accuracy: {final_acc:.4f}")
    logger.info(f"  Final validation accuracy: {final_val_acc:.4f}")
    logger.info(f"  Best validation accuracy: {max(history_dict['val_accuracy']):.4f}")
    logger.info("="*70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("="*70)
    
    return model, Xte, yte, history_dict

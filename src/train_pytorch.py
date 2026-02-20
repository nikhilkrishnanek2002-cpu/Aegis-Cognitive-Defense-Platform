import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from src.feature_extractor import get_all_features
from src.signal_generator import generate_radar_signal
from src.model_pytorch import build_pytorch_model
import cv2
import os
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Any, Optional


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
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    logger = logging.getLogger('train_pytorch')
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


class CheckpointManager:
    """Manages model checkpoints during training."""
    
    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize checkpoint manager.
        
        Parameters
        ----------
        output_dir : str
            Directory to save checkpoints
        logger : logging.Logger, optional
            Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger('train_pytorch')
        os.makedirs(output_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def save_checkpoint(self, model: nn.Module, epoch: int, loss: float,
                       is_best: bool = False, is_last: bool = False) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        model : nn.Module
            Model to save
        epoch : int
            Current epoch
        loss : float
            Current loss value
        is_best : bool
            Whether this is the best model so far
        is_last : bool
            Whether this is the last epoch
        """
        if is_best:
            self.best_loss = loss
            self.best_epoch = epoch
            path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
            }, path)
            self.logger.info(f"✓ Saved best model (loss: {loss:.4f})")
        
        if is_last:
            path = os.path.join(self.output_dir, 'last_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
            }, path)
            self.logger.info(f"✓ Saved last model (epoch: {epoch})")


def create_pytorch_dataset(samples_per_class=50):
    classes = ["drone", "aircraft", "bird", "helicopter", "missile", "clutter"]
    rd_list, spec_list, meta_list, y_list = [], [], [], []

    print("Generating simulated photonic radar dataset...")
    for label, cls in enumerate(classes):
        for _ in range(samples_per_class):
            sig = generate_radar_signal(cls)
            rd, spec, meta, _ = get_all_features(sig)
            
            # Resize to match model input
            rd = cv2.resize(rd, (128, 128))
            spec = cv2.resize(spec, (128, 128))
            
            # Normalize
            rd = rd / (np.max(rd) + 1e-8)
            spec = spec / (np.max(spec) + 1e-8)
            
            rd_list.append(rd)
            spec_list.append(spec)
            meta_list.append(meta)
            y_list.append(label)

    return (
        torch.tensor(np.array(rd_list), dtype=torch.float32),
        torch.tensor(np.array(spec_list), dtype=torch.float32),
        torch.tensor(np.array(meta_list), dtype=torch.float32),
        torch.tensor(np.array(y_list), dtype=torch.long)
    )

def train_pytorch_model(epochs: int = 10, 
                       batch_size: int = 16,
                       learning_rate: float = 0.001,
                       samples_per_class: int = 50,
                       output_dir: str = "results",
                       seed: int = 42,
                       device: Optional[str] = None,
                       dataset: Optional[Dataset] = None,
                       val_dataset: Optional[Dataset] = None) -> Tuple[Any, Dict]:
    """
    Train PyTorch model with full reproducibility and logging.
    
    Parameters
    ----------
    epochs : int, default=10
        Number of training epochs
    batch_size : int, default=16
        Batch size for training
    learning_rate : float, default=0.001
        Optimizer learning rate
    samples_per_class : int, default=50
        Samples per class in synthetic dataset
    output_dir : str, default="results"
        Directory to save checkpoints and logs
    seed : int, default=42
        Random seed for reproducibility
    device : str, optional
        Device to use ('cuda' or 'cpu'). Auto-detected if None.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset, optional
        Preconstructed dataset. When provided, the data-generation step is skipped
        and the supplied dataset is used directly. This preserves backward
        compatibility with earlier API usage.
    val_dataset : torch.utils.data.Dataset, optional
        Validation dataset used for epoch-level evaluation. When supplied the
        returned history dictionary will include `val_loss` and `val_accuracy`.

    Returns
    -------
    model : nn.Module
        Trained model
    history : dict
        Training history with keys:
        - 'loss': List of epoch losses
        - 'epoch': List of epoch numbers
        - 'lr': Learning rate used
        - 'batch_size': Batch size used
        - 'epochs': Total epochs trained
        - 'seed': Random seed used
        - 'best_loss': Best loss achieved
        - 'best_epoch': Epoch with best loss
    """
    # =====================================================================
    # SETUP PHASE
    # =====================================================================
    
    # Set reproducibility
    set_seeds(seed)
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*70)
    logger.info("🚀 STARTING REPRODUCIBLE TRAINING")
    logger.info("="*70)
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Samples per class: {samples_per_class}")
    logger.info(f"  Random seed: {seed}")
    logger.info(f"  Device: {device}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(output_dir, logger)
    
    # =====================================================================
    # DATA LOADING PHASE
    # =====================================================================
    
    logger.info("📊 Creating dataset...")
    if dataset is None:
        rd, spec, meta, y = create_pytorch_dataset(samples_per_class)
        dataset = TensorDataset(rd, spec, meta, y)
    else:
        logger.info("✓ Using externally supplied dataset")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"✓ Dataset created: {len(dataset)} samples, {len(loader)} batches")
    
    # =====================================================================
    # MODEL & OPTIMIZER PHASE
    # =====================================================================
    
    logger.info("🧠 Building model...")
    model = build_pytorch_model(num_classes=6)
    model.to(device)
    logger.info(f"✓ Model loaded on device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"✓ Optimizer: Adam (lr={learning_rate})")
    
    # =====================================================================
    # TRAINING PHASE
    # =====================================================================
    
    logger.info("📈 Starting training...")
    logger.info("-"*70)
    
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': [],
        'epoch': [],
        'lr': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'seed': seed,
        'best_loss': float('inf'),
        'best_epoch': 0,
        'device': device,
        'samples_per_class': samples_per_class,
        'timestamp': datetime.now().isoformat(),
    }
    
    model.train()
    def _evaluate_loader(eval_loader):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for b_rd, b_spec, b_meta, b_y in eval_loader:
                b_rd = b_rd.to(device)
                b_spec = b_spec.to(device)
                b_meta = b_meta.to(device)
                b_y = b_y.to(device)
                outputs = model(b_rd, b_spec, b_meta)
                loss = criterion(outputs, b_y)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == b_y).sum().item()
                total_samples += b_y.size(0)
        model.train()
        avg_loss = total_loss / max(1, len(eval_loader))
        accuracy = total_correct / max(1, total_samples)
        return avg_loss, accuracy

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = 0
        
        for batch_idx, (b_rd, b_spec, b_meta, b_y) in enumerate(loader):
            # Move to device
            b_rd = b_rd.to(device)
            b_spec = b_spec.to(device)
            b_meta = b_meta.to(device)
            b_y = b_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(b_rd, b_spec, b_meta)
            loss = criterion(outputs, b_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == b_y).sum().item()
            running_total += b_y.size(0)
            num_batches += 1
        
        # Epoch statistics
        epoch_loss = running_loss / num_batches
        train_acc = running_correct / max(1, running_total)
        history['loss'].append(epoch_loss)
        history['accuracy'].append(train_acc)
        history['epoch'].append(epoch + 1)
        
        # Check if best loss
        is_best = epoch_loss < history['best_loss']
        if is_best:
            history['best_loss'] = epoch_loss
            history['best_epoch'] = epoch + 1
        
        # Validation metrics
        if val_loader is not None:
            val_loss, val_acc = _evaluate_loader(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
        else:
            val_loss = None
            val_acc = None

        # Logging
        progress = f"Epoch [{epoch+1:3d}/{epochs}]"
        loss_str = f"Loss: {epoch_loss:.4f}"
        acc_str = f"Acc: {train_acc:.3f}"
        status = "✓ BEST" if is_best else ""
        if val_loss is not None:
            logger.info(f"{progress} | {loss_str} | {acc_str} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.3f} {status}")
        else:
            logger.info(f"{progress} | {loss_str} | {acc_str} {status}")
        
        # Save checkpoints
        checkpoint_manager.save_checkpoint(
            model, epoch + 1, epoch_loss,
            is_best=is_best,
            is_last=(epoch == epochs - 1)
        )
    
    logger.info("-"*70)
    
    # =====================================================================
    # FINALIZATION PHASE
    # =====================================================================
    
    # Save training history to JSON
    history_file = os.path.join(output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"✓ Training history saved to {history_file}")
    
    # Log summary
    logger.info("📊 Training Summary:")
    logger.info(f"  Total epochs: {history['epochs']}")
    logger.info(f"  Final loss: {history['loss'][-1]:.4f}")
    logger.info(f"  Best loss: {history['best_loss']:.4f} (epoch {history['best_epoch']})")
    logger.info(f"  Loss improvement: {(history['loss'][0] - history['best_loss']):.4f}")
    
    logger.info("="*70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("="*70)
    
    return model, history


if __name__ == "__main__":
    # Train with reproducible settings
    model, history = train_pytorch_model(
        epochs=10,
        batch_size=16,
        learning_rate=0.001,
        samples_per_class=50,
        output_dir="results",
        seed=42
    )
    
    # Plot training history (optional)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(history['epoch'], history['loss'], 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/training_loss.png', dpi=300)
        plt.close()
        print("✓ Training loss plot saved to results/training_loss.png")
    except ImportError:
        print("matplotlib not available, skipping plot")

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
from sklearn.preprocessing import StandardScaler


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
            self.logger.info(f"Saved best model (loss: {loss:.4f})")
        
        if is_last:
            path = os.path.join(self.output_dir, 'last_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
            }, path)
            self.logger.info(f"Saved last model (epoch: {epoch})")


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
            
            # Normalize images to [0, 1]
            rd = rd / (np.max(rd) + 1e-8)
            spec = spec / (np.max(spec) + 1e-8)
            
            rd_list.append(rd)
            spec_list.append(spec)
            meta_list.append(meta)
            y_list.append(label)
    
    # Convert to numpy arrays first
    rd_array = np.array(rd_list, dtype=np.float32)
    spec_array = np.array(spec_list, dtype=np.float32)
    meta_array = np.array(meta_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.long)
    
    # ⭐ CRITICAL FIX: Normalize metadata features (StandardScaler)
    # Metadata has huge scale differences (chirp_slope ~1e10 vs coherence ~0-1)
    scaler = StandardScaler()
    meta_array = scaler.fit_transform(meta_array)
    meta_array = meta_array.astype(np.float32)

    return (
        torch.tensor(rd_array, dtype=torch.float32),
        torch.tensor(spec_array, dtype=torch.float32),
        torch.tensor(meta_array, dtype=torch.float32),
        torch.tensor(y_array, dtype=torch.long)
    )


def _mixup_batch(b_rd, b_spec, b_meta, b_y, alpha: float):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(b_rd.size(0), device=b_rd.device)
    rd_mix = lam * b_rd + (1.0 - lam) * b_rd[index]
    spec_mix = lam * b_spec + (1.0 - lam) * b_spec[index]
    meta_mix = lam * b_meta + (1.0 - lam) * b_meta[index]
    y_a = b_y
    y_b = b_y[index]
    return rd_mix, spec_mix, meta_mix, y_a, y_b, lam


def _rand_bbox(h: int, w: int, lam: float):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    return x1, y1, x2, y2


def _cutmix_batch(b_rd, b_spec, b_meta, b_y, alpha: float):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(b_rd.size(0), device=b_rd.device)
    y_a = b_y
    y_b = b_y[index]

    rd_mix = b_rd.clone()
    spec_mix = b_spec.clone()
    meta_mix = b_meta.clone()

    h = b_rd.shape[-2]
    w = b_rd.shape[-1]
    x1, y1, x2, y2 = _rand_bbox(h, w, lam)

    rd_mix[..., y1:y2, x1:x2] = b_rd[index, ..., y1:y2, x1:x2]
    spec_mix[..., y1:y2, x1:x2] = b_spec[index, ..., y1:y2, x1:x2]
    meta_mix = lam * b_meta + (1.0 - lam) * b_meta[index]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
    return rd_mix, spec_mix, meta_mix, y_a, y_b, lam


def _mixup_loss(criterion, outputs, y_a, y_b, lam: float):
    return lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)

def train_pytorch_model(epochs: int = 10, 
                       batch_size: int = 16,
                       learning_rate: float = 0.001,
                       samples_per_class: int = 50,
                       output_dir: str = "results",
                       seed: int = 42,
                       device: Optional[str] = None,
                       dataset: Optional[Dataset] = None,
                       val_dataset: Optional[Dataset] = None,
                       class_weights: Optional[torch.Tensor] = None,
                       weight_decay: float = 0.0,
                       lr_schedule: Optional[Dict[str, Any]] = None,
                       label_smoothing: float = 0.0,
                       early_stopping_patience: int = 0,
                       early_stopping_min_delta: float = 0.0,
                       restore_best: bool = True,
                       mixup_alpha: float = 0.0,
                       cutmix_alpha: float = 0.0,
                       cutmix_prob: float = 0.5) -> Tuple[Any, Dict]:
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
    logger.info("STARTING REPRODUCIBLE TRAINING")
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
    
    logger.info("Creating dataset...")
    if dataset is None:
        rd, spec, meta, y = create_pytorch_dataset(samples_per_class)
        dataset = TensorDataset(rd, spec, meta, y)
    else:
        logger.info("Using externally supplied dataset")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Dataset created: {len(dataset)} samples, {len(loader)} batches")
    
    # =====================================================================
    # MODEL & OPTIMIZER PHASE
    # =====================================================================
    
    logger.info("Building model...")
    model = build_pytorch_model(num_classes=6)
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    logger.info(
        f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay}, label_smoothing={label_smoothing})"
    )

    scheduler = None
    if lr_schedule is not None:
        schedule_type = lr_schedule.get("type", "step")
        if schedule_type == "step":
            step_size = int(lr_schedule.get("step_size", 10))
            gamma = float(lr_schedule.get("gamma", 0.5))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"LR scheduler: StepLR(step_size={step_size}, gamma={gamma})")
    
    # =====================================================================
    # TRAINING PHASE
    # =====================================================================
    
    logger.info("Starting training...")
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
        'early_stopped': False,
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

    epochs_no_improve = 0

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
            
            # Forward pass with optional MixUp/CutMix
            optimizer.zero_grad()
            use_cutmix = cutmix_alpha > 0.0 and np.random.rand() < cutmix_prob
            use_mixup = mixup_alpha > 0.0 and not use_cutmix

            if use_cutmix:
                rd_mix, spec_mix, meta_mix, y_a, y_b, lam = _cutmix_batch(
                    b_rd, b_spec, b_meta, b_y, cutmix_alpha
                )
                outputs = model(rd_mix, spec_mix, meta_mix)
                loss = _mixup_loss(criterion, outputs, y_a, y_b, lam)
            elif use_mixup:
                rd_mix, spec_mix, meta_mix, y_a, y_b, lam = _mixup_batch(
                    b_rd, b_spec, b_meta, b_y, mixup_alpha
                )
                outputs = model(rd_mix, spec_mix, meta_mix)
                loss = _mixup_loss(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(b_rd, b_spec, b_meta)
                loss = criterion(outputs, b_y)
            
            # Backward pass
            loss.backward()
            
            # ⭐ CRITICAL FIX: Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Validation metrics
        if val_loader is not None:
            val_loss, val_acc = _evaluate_loader(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
        else:
            val_loss = None
            val_acc = None

        # Check if best loss (prefer validation loss when available)
        compare_loss = val_loss if val_loss is not None else epoch_loss
        is_best = compare_loss < (history['best_loss'] - early_stopping_min_delta)
        if is_best:
            history['best_loss'] = compare_loss
            history['best_epoch'] = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Logging
        progress = f"Epoch [{epoch+1:3d}/{epochs}]"
        loss_str = f"Loss: {epoch_loss:.4f}"
        acc_str = f"Acc: {train_acc:.3f}"
        status = "BEST" if is_best else ""
        if val_loss is not None:
            logger.info(f"{progress} | {loss_str} | {acc_str} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.3f} {status}")
        else:
            logger.info(f"{progress} | {loss_str} | {acc_str} {status}")
        
        if scheduler is not None:
            scheduler.step()

        # Save checkpoints
        checkpoint_manager.save_checkpoint(
            model, epoch + 1, epoch_loss,
            is_best=is_best,
            is_last=(epoch == epochs - 1)
        )

        if val_loader is not None and early_stopping_patience > 0:
            if epochs_no_improve >= early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} (no val improvement for {early_stopping_patience} epochs)"
                )
                history['early_stopped'] = True
                break
    
    logger.info("-"*70)
    
    # =====================================================================
    # FINALIZATION PHASE
    # =====================================================================
    
    # Restore best checkpoint if requested
    if restore_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        if os.path.exists(best_path):
            best_state = torch.load(best_path, map_location=device)
            model.load_state_dict(best_state['model_state_dict'])
            logger.info(f"Restored best model from {best_path}")

    # Save training history to JSON
    history_file = os.path.join(output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_file}")
    
    # Log summary
    logger.info("Training Summary:")
    logger.info(f"  Total epochs: {history['epochs']}")
    logger.info(f"  Final loss: {history['loss'][-1]:.4f}")
    logger.info(f"  Best loss: {history['best_loss']:.4f} (epoch {history['best_epoch']})")
    logger.info(f"  Loss improvement: {(history['loss'][0] - history['best_loss']):.4f}")
    
    logger.info("="*70)
    logger.info("TRAINING COMPLETE")
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

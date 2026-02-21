#!/usr/bin/env python3
"""
Experiment Runner for Cognitive Radar AI
Orchestrates data preprocessing, model training, evaluation, and result logging
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    print("⚠️  PyTorch not installed. This experiment runner requires PyTorch.")
    print("Install with: pip install torch torchvision torchaudio")
    sys.exit(1)

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_pytorch import create_pytorch_dataset
from model_pytorch import build_pytorch_model, PhotonicRadarAI
from logger import init_logging


class ExperimentRunner:
    """Main experiment orchestrator"""
    
    def __init__(self, config_path: str):
        """Initialize experiment runner with config"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = None
        self.device = None
        self.output_dirs = {}
        self.metrics = {}
        self.start_time = datetime.now()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _set_seeds(self, seed: int = None) -> None:
        """Set global random seeds for reproducibility"""
        if seed is None:
            seed = self.config.get('experiment', {}).get('seed', 42)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"[SEEDS] Set random seed to {seed}")
    
    def _create_output_dirs(self) -> Dict[str, str]:
        """Create output directory structure"""
        base_output = self.config.get('experiment', {}).get('output_dir', 'outputs')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(base_output, f"exp_{timestamp}")
        
        subdirs = {
            'base': exp_dir,
            'models': os.path.join(exp_dir, 'models'),
            'logs': os.path.join(exp_dir, 'logs'),
            'plots': os.path.join(exp_dir, 'plots'),
            'reports': os.path.join(exp_dir, 'reports'),
        }
        
        for key, path in subdirs.items():
            os.makedirs(path, exist_ok=True)
            print(f"[DIRS] Created directory: {path}")
        
        return subdirs
    
    def _setup_logging(self) -> logging.Logger:
        """Initialize logging to console and file"""
        log_config = self.config.get('logging', {})
        log_file = os.path.join(self.output_dirs['logs'], 'experiment.log')
        
        logger = logging.getLogger('experiment')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"Experiment started at {self.start_time}")
        logger.info(f"Config loaded from: {self.config_path}")
        
        return logger
    
    def _setup_device(self) -> torch.device:
        """Setup CUDA device if available"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        
        return device
    
    def _preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run data preprocessing"""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 1: DATA PREPROCESSING")
        self.logger.info("=" * 50)
        
        samples_per_class = self.config.get('experiment', {}).get('samples_per_class', 50)
        self.logger.info(f"Creating dataset with {samples_per_class} samples per class")
        
        rd, spec, meta, y = create_pytorch_dataset(samples_per_class=samples_per_class)
        
        self.logger.info(f"Range-Doppler shape: {rd.shape}")
        self.logger.info(f"Spectrogram shape: {spec.shape}")
        self.logger.info(f"Metadata shape: {meta.shape}")
        self.logger.info(f"Labels shape: {y.shape}")
        
        return rd, spec, meta, y
    
    def _train_model(
        self, 
        rd: torch.Tensor, 
        spec: torch.Tensor, 
        meta: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.nn.Module, Dict[str, list]]:
        """Run model training"""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 2: MODEL TRAINING")
        self.logger.info("=" * 50)
        
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 20)
        batch_size = training_config.get('batch_size', 16)
        learning_rate = training_config.get('learning_rate', 0.001)
        
        # Create dataset and dataloader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(rd, spec, meta, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Build model
        num_classes = self.config.get('model_config', {}).get('num_classes', 6)
        model = build_pytorch_model(num_classes=num_classes)
        model = model.to(self.device)
        
        self.logger.info(f"Model architecture: {model.__class__.__name__}")
        self.logger.info(f"Training epochs: {epochs}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Learning rate: {learning_rate}")
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        history = {
            'loss': [],
            'epoch': []
        }
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            
            for batch_idx, (b_rd, b_spec, b_meta, b_y) in enumerate(loader):
                b_rd = b_rd.to(self.device)
                b_spec = b_spec.to(self.device)
                b_meta = b_meta.to(self.device)
                b_y = b_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(b_rd, b_spec, b_meta)
                loss = criterion(outputs, b_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            
            avg_loss = running_loss / batch_count
            history['loss'].append(avg_loss)
            history['epoch'].append(epoch + 1)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.logger.info(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.logger.info("Training completed")
        
        return model, history
    
    def _evaluate_model(
        self, 
        model: torch.nn.Module,
        rd: torch.Tensor,
        spec: torch.Tensor,
        meta: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, Any]:
        """Run model evaluation"""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 3: MODEL EVALUATION")
        self.logger.info("=" * 50)
        
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        dataset = TensorDataset(rd, spec, meta, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate
        model.eval()
        all_preds = []
        all_y = []
        
        with torch.no_grad():
            for b_rd, b_spec, b_meta, b_y in loader:
                b_rd = b_rd.to(self.device)
                b_spec = b_spec.to(self.device)
                b_meta = b_meta.to(self.device)
                
                outputs = model(b_rd, b_spec, b_meta)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_y.extend(b_y.numpy())
        
        all_preds = np.array(all_preds)
        all_y = np.array(all_y)
        
        # Compute metrics
        accuracy = accuracy_score(all_y, all_preds)
        cm = confusion_matrix(all_y, all_preds)
        
        # Detection metrics
        Pd = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
        FAR = (np.sum(cm) - np.trace(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0.0
        
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Probability of Detection (Pd): {Pd:.4f}")
        self.logger.info(f"False Alarm Rate (FAR): {FAR:.4f}")
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(self.output_dirs['plots'], 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        self.logger.info(f"Saved confusion matrix to: {cm_path}")
        
        metrics = {
            'accuracy': float(accuracy),
            'probability_of_detection': float(Pd),
            'false_alarm_rate': float(FAR),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def _save_results(
        self, 
        model: torch.nn.Module,
        history: Dict[str, list],
        metrics: Dict[str, Any]
    ) -> None:
        """Save trained model and results"""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 4: SAVING RESULTS")
        self.logger.info("=" * 50)
        
        # Save model
        model_path = os.path.join(self.output_dirs['models'], 'model_final.pt')
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Saved model to: {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.output_dirs['reports'], 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved metrics to: {metrics_path}")
        
        # Save training history
        history_path = os.path.join(self.output_dirs['reports'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"Saved training history to: {history_path}")
        
        # Save config copy
        config_path = os.path.join(self.output_dirs['reports'], 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        self.logger.info(f"Saved config to: {config_path}")
    
    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print experiment summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info("=" * 50)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total duration: {duration}")
        self.logger.info("")
        self.logger.info("Performance Metrics:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Probability of Detection: {metrics['probability_of_detection']:.4f}")
        self.logger.info(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
        self.logger.info("")
        self.logger.info(f"Output directory: {self.output_dirs['base']}")
        self.logger.info("")
        
        # Print to console as well
        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print("")
        print("Performance Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Probability of Detection: {metrics['probability_of_detection']:.4f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
        print("")
        print(f"Output directory: {self.output_dirs['base']}")
        print("=" * 50 + "\n")
    
    def run(self) -> None:
        """Execute the full experiment pipeline"""
        try:
            # Setup
            self._set_seeds()
            self.output_dirs = self._create_output_dirs()
            self.logger = self._setup_logging()
            self.device = self._setup_device()
            
            self.logger.info("Starting experiment pipeline...")
            
            # Pipeline stages
            rd, spec, meta, y = self._preprocess_data()
            model, history = self._train_model(rd, spec, meta, y)
            metrics = self._evaluate_model(model, rd, spec, meta, y)
            self._save_results(model, history, metrics)
            self._print_summary(metrics)
            
            self.logger.info("Experiment completed successfully!")
            
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Experiment failed with error: {str(e)}")
            print(f"ERROR: Experiment failed - {str(e)}", file=sys.stderr)
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Experiment Runner for Cognitive Radar AI'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to experiment config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    runner.run()


if __name__ == '__main__':
    main()

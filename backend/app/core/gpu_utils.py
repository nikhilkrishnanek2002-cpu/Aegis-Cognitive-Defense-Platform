"""GPU utilities - safe CUDA detection and management."""

import torch
import os
from app.core.logging import pipeline_logger


class GPUManager:
    """Manages GPU/CUDA configuration safely."""
    
    def __init__(self):
        self.cuda_available = False
        self.device = "cpu"
        self.gpu_count = 0
        self.initialize()
    
    def initialize(self):
        """Initialize GPU configuration."""
        try:
            # Check if CUDA is available
            self.cuda_available = torch.cuda.is_available()
            self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0
            
            # Get device preference from config
            from app.core.config import get_config
            config = get_config()
            device_pref = config.model_device.lower()
            
            # Determine final device
            if device_pref == "auto":
                # Auto-detect: use GPU if available, fallback to CPU
                if self.cuda_available and self.gpu_count > 0:
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            elif device_pref == "cuda":
                # Force CUDA - will log warning if unavailable
                if self.cuda_available and self.gpu_count > 0:
                    self.device = "cuda"
                else:
                    pipeline_logger.warning("CUDA requested but not available - falling back to CPU")
                    self.device = "cpu"
            else:
                # CPU requested
                self.device = "cpu"
            
            if self.device == "cuda" and self.cuda_available and self.gpu_count > 0:
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                pipeline_logger.info("=" * 70)
                pipeline_logger.info("🎮 GPU ACCELERATION ENABLED")
                pipeline_logger.info("=" * 70)
                pipeline_logger.info(f"✓ CUDA Available: {self.cuda_available}")
                pipeline_logger.info(f"✓ GPU Count: {self.gpu_count}")
                pipeline_logger.info(f"✓ GPU Name: {gpu_name}")
                pipeline_logger.info(f"✓ GPU Memory: {gpu_memory:.2f} GB")
                pipeline_logger.info(f"✓ Device: {self.device}")
                pipeline_logger.info("=" * 70)
            else:
                pipeline_logger.info("⚠ GPU not available - using CPU (slower)")
                pipeline_logger.info(f"  CUDA Available: {self.cuda_available}")
                pipeline_logger.info(f"  Device: cpu")
        
        except Exception as e:
            self.device = "cpu"
            pipeline_logger.warning(f"GPU initialization failed: {e}")
            pipeline_logger.info("  Falling back to CPU")
    
    def get_device(self) -> str:
        """Get current device ('cuda' or 'cpu')."""
        return self.device
    
    def get_torch_device(self):
        """Get PyTorch device object."""
        return torch.device(self.device)
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.cuda_available and self.gpu_count > 0
    
    def print_status(self):
        """Print GPU status."""
        status = "✅ GPU Ready" if self.is_gpu_available() else "❌ CPU Only"
        pipeline_logger.info(f"\n{status}")
        pipeline_logger.info(f"  Device: {self.device}")
        pipeline_logger.info(f"  GPUs: {self.gpu_count}")


# Global GPU manager instance
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """Get or create GPU manager singleton."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def get_device() -> str:
    """Get current device ('cuda' or 'cpu')."""
    return get_gpu_manager().get_device()


def get_torch_device():
    """Get PyTorch device object."""
    return get_gpu_manager().get_torch_device()


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return get_gpu_manager().is_gpu_available()

"""
Integration Guide: Reporting Module with Experiment Runner

This file shows how to integrate the reporting module into the 
experiment_runner.py evaluation pipeline.
"""

# ============================================================================
# EXAMPLE 1: Minimal Integration (add to experiment_runner.py)
# ============================================================================

# At the top of experiment_runner.py, add import:
# from src.reporting import (
#     plot_confusion_matrix,
#     plot_roc_curve,
#     plot_precision_recall,
#     plot_training_history,
#     plot_detection_vs_snr,
#     plot_tracking_rmse,
# )

# In the _evaluate_model() method, add after computing metrics:
"""
def _evaluate_model(self, model, test_loader, output_dir):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Compute metrics
    accuracy = (y_pred == y_true).mean()
    self.logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # ===== ADD THESE LINES FOR VISUALIZATION =====
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(plots_dir, "confusion_matrix.png")
    )
    self.logger.info(f"Saved confusion matrix to {plots_dir}")
    
    # Generate ROC curve (for binary classification)
    if len(np.unique(y_true)) == 2:
        plot_roc_curve(
            y_true, y_prob[:, 1],
            os.path.join(plots_dir, "roc_curve.png")
        )
        self.logger.info("Generated ROC curve")
    
    # Generate Precision-Recall curve
    if len(np.unique(y_true)) == 2:
        plot_precision_recall(
            y_true, y_prob[:, 1],
            os.path.join(plots_dir, "precision_recall.png")
        )
        self.logger.info("Generated Precision-Recall curve")
    
    # ===== END ADD =====
    
    return accuracy
"""

# ============================================================================
# EXAMPLE 2: Full Integration with Training History
# ============================================================================

"""
def run(self):
    '''Run the complete experiment pipeline with visualization.'''
    try:
        self._setup_logging()
        self._setup_device()
        self.logger.info("Starting experiment...")
        
        # Load config and create output directories
        config = self._load_config()
        output_dir = self._create_output_dirs()
        
        # Set seeds for reproducibility
        self._set_seeds(config.get('random_seed', 42))
        
        # Preprocess data
        self.logger.info("Preprocessing data...")
        train_data, val_data, test_data = self._preprocess_data(config)
        
        # Train model
        self.logger.info("Training model...")
        model, history = self._train_model(config, train_data, val_data, output_dir)
        
        # ===== ADD TRAINING HISTORY PLOT =====
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_training_history(
            history,
            os.path.join(plots_dir, "training_history.png")
        )
        self.logger.info("Generated training history plot")
        # ===== END ADD =====
        
        # Evaluate model
        self.logger.info("Evaluating model...")
        metrics = self._evaluate_model(model, test_data, output_dir)
        
        # Save results (includes figures)
        self._save_results(model, metrics, output_dir)
        self._print_summary(metrics, output_dir)
        
        self.logger.info("✅ Experiment completed successfully")
        
    except Exception as e:
        self.logger.error(f"❌ Experiment failed: {str(e)}", exc_info=True)
        raise
"""

# ============================================================================
# EXAMPLE 3: SNR Sweep Analysis
# ============================================================================

"""
def run_snr_sweep(self, snr_range=(-5, 20), num_points=11):
    '''Run experiment across different SNR levels and plot results.'''
    from src.reporting import plot_detection_vs_snr
    
    snr_values = np.linspace(snr_range[0], snr_range[1], num_points)
    accuracies = []
    
    for snr in snr_values:
        self.logger.info(f"Testing SNR = {snr} dB...")
        
        config = self._load_config()
        config['dataset']['target_snr_db'] = snr
        
        train_data, val_data, test_data = self._preprocess_data(config)
        model, _ = self._train_model(config, train_data, val_data, "temp")
        acc = self._evaluate_model(model, test_data, "temp")
        
        accuracies.append(acc)
        self.logger.info(f"  Accuracy at SNR={snr} dB: {acc:.4f}")
    
    # Plot results
    output_dir = self._create_output_dirs()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_detection_vs_snr(
        snr_values, np.array(accuracies),
        os.path.join(plots_dir, "detection_vs_snr_sweep.png")
    )
    
    self.logger.info(f"✅ SNR sweep completed. Plot saved to {plots_dir}")
    return snr_values, accuracies
"""

# ============================================================================
# EXAMPLE 4: Standalone Usage
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from reporting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_training_history,
    plot_detection_vs_snr,
    plot_tracking_rmse,
)


def generate_all_plots_from_experiment(experiment_results_dir):
    """
    Generate all plots from a completed experiment's results.
    
    Expected directory structure:
        experiment_results/
        ├── metrics.json
        ├── training_history.json
        └── plots/
    """
    
    import json
    
    plots_dir = os.path.join(experiment_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load metrics
    with open(os.path.join(experiment_results_dir, "metrics.json")) as f:
        metrics = json.load(f)
    
    # Load training history
    with open(os.path.join(experiment_results_dir, "training_history.json")) as f:
        history = json.load(f)
    
    # Generate plots if data available
    if 'confusion_matrix' in metrics:
        y_true = np.array(metrics['y_true'])
        y_pred = np.array(metrics['y_pred'])
        plot_confusion_matrix(
            y_true, y_pred,
            os.path.join(plots_dir, "confusion_matrix.png")
        )
        print("✅ Generated confusion_matrix.png")
    
    if 'roc_curve_data' in metrics:
        y_true = np.array(metrics['y_true'])
        y_prob = np.array(metrics['y_prob'])
        plot_roc_curve(
            y_true, y_prob,
            os.path.join(plots_dir, "roc_curve.png")
        )
        print("✅ Generated roc_curve.png")
    
    if history:
        plot_training_history(
            history,
            os.path.join(plots_dir, "training_history.png")
        )
        print("✅ Generated training_history.png")
    
    return plots_dir


# ============================================================================
# EXAMPLE 5: Quick Test in Jupyter Notebook
# ============================================================================

"""
# In Jupyter:
%matplotlib inline
import sys, os
sys.path.insert(0, 'src')
from reporting import plot_confusion_matrix
import numpy as np

# Quick test
y_true = np.array([0, 1, 2, 0, 1, 2] * 5)
y_pred = np.array([0, 1, 1, 0, 1, 2] * 5)

plot_confusion_matrix(y_true, y_pred, "test_cm.png")
# Figure saved! View it:
# from IPython.display import Image
# Image("test_cm.png")
"""

# ============================================================================
# EXAMPLE 6: Configuration-Driven Reporting
# ============================================================================

"""
# In config.yaml, add:
reporting:
  enabled: true
  output_format: "png"  # png, pdf, or eps
  dpi: 300
  plots:
    - confusion_matrix
    - roc_curve
    - precision_recall
    - training_history
    - detection_vs_snr

# In experiment_runner.py:
def _should_generate_plot(self, plot_name):
    reporting_config = self.config.get('reporting', {})
    if not reporting_config.get('enabled', True):
        return False
    return plot_name in reporting_config.get('plots', [])

def _get_output_format(self):
    reporting_config = self.config.get('reporting', {})
    ext = reporting_config.get('output_format', 'png')
    return '.' + ext if not ext.startswith('.') else ext
"""

# ============================================================================
# QUICK REFERENCE: Import Patterns
# ============================================================================

# Import all functions
# from src.reporting import *

# Import specific functions
# from src.reporting import plot_confusion_matrix, plot_roc_curve

# Import with alias
# from src import reporting
# reporting.plot_confusion_matrix(y_true, y_pred, save_path)

# ============================================================================
# INTEGRATION CHECKLIST
# ============================================================================

INTEGRATION_CHECKLIST = """
☐ 1. Add import statement at top of experiment_runner.py
☐ 2. Add plots_dir = os.path.join(output_dir, "plots")
☐ 3. Call plot_* functions in evaluation phase
☐ 4. Verify output directory exists before saving
☐ 5. Add logging statements for plot generation
☐ 6. Test with example data using examples_reporting.py
☐ 7. View generated plots in results/reports/
☐ 8. Customize plot format (PNG/PDF/EPS) as needed
☐ 9. Add reporting section to YAML config (optional)
☐ 10. Document in experiment summary
"""

print(INTEGRATION_CHECKLIST)

# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

PERFORMANCE_TIPS = """
PERFORMANCE OPTIMIZATION:

1. BATCH PLOTTING
   - Generate multiple plots in a loop
   - Already fast (<1s per plot)
   
2. FORMAT SELECTION
   - PNG: Good for presentations, web, fast generation
   - PDF: Best for LaTeX/academic papers, slow (5-10s)
   - EPS: For publishing, slow (5-10s)
   
3. MEMORY USAGE
   - Plots saved directly to disk
   - No memory accumulation in loops
   - Safe for batch processing
   
4. PARALLEL PROCESSING
   - Can generate plots in separate threads
   - Consider for very large experiments
   
5. CACHING
   - Save metrics to JSON
   - Regenerate plots without retraining
   - Use generate_all_plots_from_experiment()
"""

print(PERFORMANCE_TIPS)

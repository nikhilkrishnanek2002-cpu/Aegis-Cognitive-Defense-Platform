"""
Enhanced Evaluation Pipeline - Integration Examples

Demonstrates how to use the enhanced evaluation module with:
  1. PyTorch models
  2. Keras/TensorFlow models
  3. Integration with the reporting module for visualization
  4. Metrics comparison between models
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def example_1_basic_evaluation():
    """
    Example 1: Basic evaluation with synthetic predictions
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: BASIC COMPREHENSIVE EVALUATION")
    print("=" * 80)

    from src.evaluation_enhanced import compute_comprehensive_metrics, get_metrics_summary

    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 3

    # Create realistic predictions
    y_true = np.array([0] * 35 + [1] * 33 + [2] * 32)
    np.random.shuffle(y_true)

    # Simulate model predictions (>80% accuracy)
    y_pred = y_true.copy()
    noise_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    for idx in noise_indices:
        y_pred[idx] = (y_true[idx] + np.random.randint(1, n_classes)) % n_classes

    # Generate probability predictions (convert to softmax-like)
    y_probs = np.random.rand(n_samples, n_classes)
    for i in range(n_samples):
        # Boost probability of correct class
        y_probs[i, y_pred[i]] += 1.0
        y_probs[i] = y_probs[i] / y_probs[i].sum()

    print("\n📊 Evaluating model predictions...")
    print(f"   Samples: {n_samples}, Classes: {n_classes}")

    # Compute metrics
    metrics = compute_comprehensive_metrics(
        predictions=y_pred,
        labels=y_true,
        probabilities=y_probs,
        output_dir="outputs/reports",
        model_name="radar_classifier_v1",
    )

    # Display summary
    summary = get_metrics_summary(metrics)
    print(summary)

    return metrics


def example_2_model_comparison():
    """
    Example 2: Compare two models
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MODEL COMPARISON")
    print("=" * 80)

    from src.evaluation_enhanced import (
        compute_comprehensive_metrics,
        get_metrics_summary,
        compare_metrics,
    )

    np.random.seed(42)
    n_samples = 200
    n_classes = 3

    # Ground truth
    y_true = np.tile([0, 1, 2], n_samples // 3 + 1)[:n_samples]

    print("\n📊 Training Model 1 (Baseline)...")
    # Model 1: baseline model (~78% accuracy)
    y_pred1 = y_true.copy()
    noise_idx1 = np.random.choice(n_samples, size=int(0.22 * n_samples), replace=False)
    for idx in noise_idx1:
        y_pred1[idx] = (y_true[idx] + np.random.randint(1, n_classes)) % n_classes

    y_probs1 = np.random.rand(n_samples, n_classes)
    for i in range(n_samples):
        y_probs1[i, y_pred1[i]] += 0.8
        y_probs1[i] = y_probs1[i] / y_probs1[i].sum()

    metrics1 = compute_comprehensive_metrics(
        predictions=y_pred1,
        labels=y_true,
        probabilities=y_probs1,
        output_dir="outputs/reports",
        model_name="radar_classifier_baseline",
    )

    print("📊 Training Model 2 (Improved)...")
    # Model 2: improved model (~85% accuracy)
    y_pred2 = y_true.copy()
    noise_idx2 = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    for idx in noise_idx2:
        y_pred2[idx] = (y_true[idx] + np.random.randint(1, n_classes)) % n_classes

    y_probs2 = np.random.rand(n_samples, n_classes)
    for i in range(n_samples):
        y_probs2[i, y_pred2[i]] += 1.2
        y_probs2[i] = y_probs2[i] / y_probs2[i].sum()

    metrics2 = compute_comprehensive_metrics(
        predictions=y_pred2,
        labels=y_true,
        probabilities=y_probs2,
        output_dir="outputs/reports",
        model_name="radar_classifier_improved",
    )

    # Compare
    print(compare_metrics(metrics1, metrics2, ("Baseline", "Improved")))

    return metrics1, metrics2


def example_3_with_reporting_module():
    """
    Example 3: Integrate with reporting module for visualization
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: INTEGRATION WITH REPORTING MODULE")
    print("=" * 80)

    try:
        from src.evaluation_enhanced import compute_comprehensive_metrics
        from src.reporting import plot_confusion_matrix, plot_training_history
    except ImportError:
        print("⚠️  Reporting module not available. Skipping visualization example.")
        return

    np.random.seed(42)
    n_samples = 150
    n_classes = 3

    # Generate data
    y_true = np.tile([0, 1, 2], n_samples // 3 + 1)[:n_samples]
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    for idx in noise_idx:
        y_pred[idx] = (y_true[idx] + np.random.randint(1, n_classes)) % n_classes

    y_probs = np.random.rand(n_samples, n_classes)
    for i in range(n_samples):
        y_probs[i, y_pred[i]] += 1.0
        y_probs[i] = y_probs[i] / y_probs[i].sum()

    # Compute metrics
    print("\n📊 Computing comprehensive metrics...")
    metrics = compute_comprehensive_metrics(
        predictions=y_pred,
        labels=y_true,
        probabilities=y_probs,
        output_dir="outputs/reports",
        model_name="radar_model_with_visualization",
    )

    print(f"✓ Metrics saved to: {metrics['_metrics_file']}")

    # Visualize confusion matrix
    print("📊 Creating confusion matrix visualization...")
    try:
        plot_confusion_matrix(
            cm=np.array(metrics["cm"]),
            class_names=[f"Class {i}" for i in range(metrics["metadata"]["n_classes"])],
            output_file="outputs/reports/confusion_matrix_report.png",
        )
        print("✓ Confusion matrix saved to: outputs/reports/confusion_matrix_report.png")
    except Exception as e:
        print(f"⚠️  Could not create confusion matrix visualization: {e}")

    # Visualize training history (if available)
    try:
        if os.path.exists("results/training_history.json"):
            with open("results/training_history.json") as f:
                history = json.load(f)
            plot_training_history(
                history=history,
                output_file="outputs/reports/training_history_report.png",
            )
            print("✓ Training history saved to: outputs/reports/training_history_report.png")
    except Exception as e:
        print(f"⚠️  Could not visualize training history: {e}")

    return metrics


def example_4_pytorch_integration():
    """
    Example 4: PyTorch model evaluation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: PYTORCH MODEL INTEGRATION")
    print("=" * 80)

    print("\n📌 Code Template for PyTorch Models:")
    template = '''
# After training your PyTorch model
from src.evaluation_enhanced import evaluate_pytorch_enhanced

# Assuming model and test_loader are defined
metrics = evaluate_pytorch_enhanced(
    model=model,
    loader=test_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="outputs/reports",
    model_name="radar_model",
)

# Access metrics
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1:  {metrics['macro_avg']['f1']:.4f}")
print(f"ROC-AUC:   {metrics.get('roc_auc_macro', 'N/A')}")

# Confusion matrix available as:
cm = metrics['cm']
cm_norm = metrics['cm_normalized']

# All metrics saved to:
# outputs/reports/metrics.json
    '''
    print(template)

    # Example: Create a dummy model scenario
    print("📌 Example Implementation:")
    print("""
# Step 1: Train model
from src.train_pytorch import train_pytorch_model

model, history = train_pytorch_model(
    epochs=20,
    seed=42,
    output_dir="results"
)

# Step 2: Evaluate on test set
from src.evaluation_enhanced import evaluate_pytorch_enhanced

test_loader = get_test_loader()  # Your data loader
metrics = evaluate_pytorch_enhanced(
    model=model,
    loader=test_loader,
    output_dir="outputs/reports",
    model_name="radar_model_v1"
)

# Step 3: Visualize results
from src.reporting import plot_confusion_matrix
plot_confusion_matrix(
    cm=metrics['cm'],
    class_names=['Class 0', 'Class 1', 'Class 2']
)
    """)


def example_5_metrics_structure():
    """
    Example 5: Explore metrics structure for custom use cases
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: METRICS STRUCTURE & CUSTOM USE CASES")
    print("=" * 80)

    from src.evaluation_enhanced import compute_comprehensive_metrics

    np.random.seed(42)
    y_true = np.array([0, 1, 2, 0, 1, 2] * 5)
    y_pred = np.array([0, 1, 2, 0, 2, 1] * 5)
    y_probs = np.random.rand(30, 3)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    metrics = compute_comprehensive_metrics(
        predictions=y_pred,
        labels=y_true,
        probabilities=y_probs,
        output_dir="outputs/reports",
        model_name="example_model",
    )

    print("\n📊 Metrics Dictionary Structure:")
    print("\nTop-level keys:")
    for key in sorted(metrics.keys()):
        if key not in ["classification_report", "_metrics_file"]:
            val = metrics[key]
            if isinstance(val, dict):
                print(f"  {key}: {{...}} (dict with keys: {list(val.keys())})")
            elif isinstance(val, list):
                print(f"  {key}: [...] (list with {len(val)} items)")
            else:
                print(f"  {key}: {type(val).__name__}")

    print("\n📌 Common Custom Use Cases:")
    print("\n1. Get best and worst performing classes:")
    f1_scores = metrics["per_class"]["f1"]
    best_class = np.argmax(f1_scores)
    worst_class = np.argmin(f1_scores)
    print(f"   Best class: {best_class} (F1={f1_scores[best_class]:.4f})")
    print(f"   Worst class: {worst_class} (F1={f1_scores[worst_class]:.4f})")

    print("\n2. Extract confusion matrix for analysis:")
    cm = np.array(metrics["cm"])
    print(f"   Shape: {cm.shape}")
    print(f"   Diagonal (correct predictions): {np.diag(cm)}")
    print(f"   Off-diagonal (misclassifications): {np.sum(cm) - np.trace(cm)}")

    print("\n3. Load metrics from saved file for comparison:")
    metrics_file = metrics["_metrics_file"]
    print(f"   Saved to: {metrics_file}")
    print(f"   Load with: metrics = load_metrics_from_file('{metrics_file}')")

    print("\n4. Generate custom reports:")
    print(f"   Dataset: {metrics['metadata']['n_samples']} samples")
    print(f"   Classes: {metrics['metadata']['n_classes']}")
    print(f"   Model: {metrics['metadata']['model_name']}")
    print(f"   Time: {metrics['metadata']['timestamp']}")

    return metrics


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "ENHANCED EVALUATION PIPELINE - INTEGRATION EXAMPLES".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Create output directory
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    # Run all examples
    print("\n" + "─" * 80)
    example_1_basic_evaluation()

    print("\n" + "─" * 80)
    example_2_model_comparison()

    print("\n" + "─" * 80)
    example_3_with_reporting_module()

    print("\n" + "─" * 80)
    example_4_pytorch_integration()

    print("\n" + "─" * 80)
    example_5_metrics_structure()

    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\n📁 Output files generated:")
    print("   ✓ outputs/reports/metrics.json - Comprehensive metrics")
    print("   ✓ outputs/reports/ - All evaluation outputs")
    print("\n📖 For more information, see: EVALUATION_ENHANCED_GUIDE.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

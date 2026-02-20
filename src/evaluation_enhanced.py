"""
Enhanced Evaluation Pipeline
=============================

Comprehensive evaluation metrics for radar AI models:
  - Accuracy, Precision, Recall, F1 Score
  - ROC-AUC scores
  - Confusion matrices
  - Per-class metrics

Saves metrics to JSON for reproducibility and further analysis.
Returns structured dictionaries for downstream plotting and reporting.

Usage:
    from src.evaluation_enhanced import compute_comprehensive_metrics
    
    # After model evaluation
    metrics = compute_comprehensive_metrics(
        predictions,
        labels,
        probabilities,  # optional, for ROC-AUC
        output_dir="outputs/reports"
    )
    
    # Access metrics
    print(metrics['accuracy'])
    print(metrics['per_class']['precision'])
    print(metrics['cm'])  # confusion matrix
"""

import json
import os
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def _ensure_output_dir(output_dir: str) -> str:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy/sklearn types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def compute_comprehensive_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    output_dir: str = "outputs/reports",
    model_name: str = "radar_model",
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for multi-class classification.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels (shape: (n_samples,), dtype: int)
    labels : np.ndarray
        Ground truth class labels (shape: (n_samples,), dtype: int)
    probabilities : Optional[np.ndarray]
        Predicted probabilities/logits (shape: (n_samples, n_classes), dtype: float)
        If provided, used for ROC-AUC computation
    output_dir : str
        Directory to save metrics JSON file to
    model_name : str
        Model identifier for logging
    num_classes : Optional[int]
        Number of classes. If None, inferred from data.

    Returns
    -------
    Dict[str, Any]
        Comprehensive metrics dictionary with structure:
        {
            'accuracy': float,
            'macro_avg': {
                'precision': float,
                'recall': float,
                'f1': float,
            },
            'weighted_avg': {
                'precision': float,
                'recall': float,
                'f1': float,
            },
            'per_class': {
                'precision': List[float],
                'recall': List[float],
                'f1': List[float],
            },
            'cm': List[List[int]],  # Confusion matrix
            'cm_normalized': List[List[float]],  # Normalized confusion matrix
            'roc_auc': Optional[float],  # Binary classification only
            'roc_auc_macro': Optional[float],  # Multi-class One-vs-Rest
            'roc_auc_weighted': Optional[float],  # Multi-class One-vs-Rest weighted
            'classification_report': Dict,  # Full sklearn report
            'metadata': {
                'n_samples': int,
                'n_classes': int,
                'model_name': str,
                'timestamp': str,
            }
        }

    Examples
    --------
    >>> # Binary classification
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> y_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2], [0.2, 0.8]])
    >>> metrics = compute_comprehensive_metrics(y_pred, y_true, y_probs)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    >>> # Multi-class classification
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 1, 2, 0, 2, 1])
    >>> y_probs = np.random.rand(6, 3)
    >>> y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize to probabilities
    >>> metrics = compute_comprehensive_metrics(y_pred, y_true, y_probs)
    >>> print(f"Macro F1: {metrics['macro_avg']['f1']:.4f}")
    """

    # Validate inputs
    assert len(predictions) == len(labels), "Predictions and labels must have same length"
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if num_classes is None:
        num_classes = max(predictions.max(), labels.max()) + 1

    # Ensure output directory exists
    output_dir = _ensure_output_dir(output_dir)

    # 1. Basic metrics
    accuracy = accuracy_score(labels, predictions)

    # 2. Per-class metrics (precision, recall, F1)
    precision_per_class = precision_score(
        labels, predictions, average=None, zero_division=0, labels=range(num_classes)
    )
    recall_per_class = recall_score(
        labels, predictions, average=None, zero_division=0, labels=range(num_classes)
    )
    f1_per_class = f1_score(
        labels, predictions, average=None, zero_division=0, labels=range(num_classes)
    )

    # 3. Aggregated metrics (macro and weighted averages)
    precision_macro = precision_score(
        labels, predictions, average="macro", zero_division=0
    )
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

    precision_weighted = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        labels, predictions, average="weighted", zero_division=0
    )
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    # 4. Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    # Handle division by zero (rows with no samples)
    cm_normalized = np.nan_to_num(cm_normalized)

    # 5. ROC-AUC computation (if probabilities provided)
    roc_auc_scores = _compute_roc_auc_scores(
        labels, probabilities, num_classes
    ) if probabilities is not None else {}

    # 6. Classification report
    class_report = classification_report(
        labels,
        predictions,
        labels=range(num_classes),
        output_dict=True,
        zero_division=0,
    )

    # 7. Metadata
    metadata = {
        "n_samples": int(len(labels)),
        "n_classes": int(num_classes),
        "model_name": str(model_name),
        "timestamp": datetime.now().isoformat(),
    }

    # 8. Assemble results dictionary with JSON-serializable types
    metrics_dict = {
        "accuracy": float(accuracy),
        "macro_avg": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro),
        },
        "weighted_avg": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1": float(f1_weighted),
        },
        "per_class": {
            "precision": [float(p) for p in precision_per_class],
            "recall": [float(r) for r in recall_per_class],
            "f1": [float(f) for f in f1_per_class],
        },
        "cm": [[int(x) for x in row] for row in cm.tolist()],
        "cm_normalized": [[float(x) for x in row] for row in cm_normalized.tolist()],
        **roc_auc_scores,
        "classification_report": _make_json_serializable(class_report),
        "metadata": metadata,
    }

    # 9. Save to JSON
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Add file path to metadata for traceability
    metrics_dict["_metrics_file"] = metrics_file

    return metrics_dict


def _compute_roc_auc_scores(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Compute ROC-AUC scores for binary and multi-class problems.

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels (shape: (n_samples,))
    probabilities : np.ndarray
        Predicted probabilities (shape: (n_samples, n_classes))
    num_classes : int
        Number of classes

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'roc_auc', 'roc_auc_macro', 'roc_auc_weighted' keys
    """
    roc_scores = {}

    try:
        if num_classes == 2:
            # Binary classification: use 1st class probabilities
            y_probs_binary = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            auc = roc_auc_score(labels, y_probs_binary)
            roc_scores["roc_auc"] = float(auc)
            roc_scores["roc_auc_macro"] = None
            roc_scores["roc_auc_weighted"] = None

        else:
            # Multi-class: One-vs-Rest
            # Ensure probabilities are normalized
            if probabilities.min() < 0 or probabilities.max() > 1:
                # Convert logits to probabilities via softmax
                exp_probs = np.exp(probabilities - probabilities.max(axis=1, keepdims=True))
                y_probs_norm = exp_probs / exp_probs.sum(axis=1, keepdims=True)
            else:
                y_probs_norm = probabilities

            auc_macro = roc_auc_score(
                labels, y_probs_norm, multi_class="ovr", average="macro"
            )
            auc_weighted = roc_auc_score(
                labels, y_probs_norm, multi_class="ovr", average="weighted"
            )
            roc_scores["roc_auc"] = None  # Not applicable for multi-class
            roc_scores["roc_auc_macro"] = float(auc_macro)
            roc_scores["roc_auc_weighted"] = float(auc_weighted)

    except Exception as e:
        # Silently skip ROC-AUC computation if it fails
        roc_scores["roc_auc"] = None
        roc_scores["roc_auc_macro"] = None
        roc_scores["roc_auc_weighted"] = None

    return roc_scores


def get_metrics_summary(metrics: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of metrics.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics dictionary from compute_comprehensive_metrics()

    Returns
    -------
    str
        Formatted summary string
    """
    summary = []
    summary.append("=" * 70)
    summary.append("EVALUATION METRICS SUMMARY")
    summary.append("=" * 70)
    summary.append(f"\nDataset: {metrics['metadata']['n_samples']} samples, {metrics['metadata']['n_classes']} classes")
    summary.append(f"Model: {metrics['metadata']['model_name']}")
    summary.append(f"Timestamp: {metrics['metadata']['timestamp']}\n")

    summary.append(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    summary.append("\nMacro-Averaged Metrics (unweighted):")
    summary.append(f"  Precision: {metrics['macro_avg']['precision']:.4f}")
    summary.append(f"  Recall:    {metrics['macro_avg']['recall']:.4f}")
    summary.append(f"  F1 Score:  {metrics['macro_avg']['f1']:.4f}")

    summary.append("\nWeighted-Averaged Metrics (by class support):")
    summary.append(f"  Precision: {metrics['weighted_avg']['precision']:.4f}")
    summary.append(f"  Recall:    {metrics['weighted_avg']['recall']:.4f}")
    summary.append(f"  F1 Score:  {metrics['weighted_avg']['f1']:.4f}")

    if metrics['metadata']['n_classes'] <= 10:
        summary.append("\nPer-Class Metrics:")
        for i in range(metrics['metadata']['n_classes']):
            prec = metrics['per_class']['precision'][i]
            rec = metrics['per_class']['recall'][i]
            f1 = metrics['per_class']['f1'][i]
            summary.append(f"  Class {i}: P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    if metrics.get('roc_auc') is not None:
        summary.append(f"\nROC-AUC (Binary): {metrics['roc_auc']:.4f}")
    if metrics.get('roc_auc_macro') is not None:
        summary.append(f"ROC-AUC Macro (Multi-class): {metrics['roc_auc_macro']:.4f}")
    if metrics.get('roc_auc_weighted') is not None:
        summary.append(f"ROC-AUC Weighted (Multi-class): {metrics['roc_auc_weighted']:.4f}")

    summary.append("\n" + "=" * 70)

    return "\n".join(summary)


def load_metrics_from_file(metrics_file: str) -> Dict[str, Any]:
    """
    Load previously saved metrics from JSON file.

    Parameters
    ----------
    metrics_file : str
        Path to metrics.json file

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary
    """
    with open(metrics_file, "r") as f:
        return json.load(f)


def compare_metrics(
    metrics1: Dict[str, Any],
    metrics2: Dict[str, Any],
    model_names: Tuple[str, str] = ("Model 1", "Model 2"),
) -> str:
    """
    Compare two metrics dictionaries and return formatted comparison.

    Parameters
    ----------
    metrics1 : Dict[str, Any]
        First metrics dictionary
    metrics2 : Dict[str, Any]
        Second metrics dictionary
    model_names : Tuple[str, str]
        Names for the two models

    Returns
    -------
    str
        Formatted comparison string
    """
    name1, name2 = model_names
    comparison = []
    comparison.append("=" * 70)
    comparison.append("METRICS COMPARISON")
    comparison.append("=" * 70)
    comparison.append(f"\n{name1:30} vs {name2:30}\n")

    # Accuracy
    acc1 = metrics1["accuracy"]
    acc2 = metrics2["accuracy"]
    delta = acc2 - acc1
    comparison.append(f"Accuracy:          {acc1:.4f}        {acc2:.4f}  (Δ: {delta:+.4f})")

    # Macro F1
    f1_mac1 = metrics1["macro_avg"]["f1"]
    f1_mac2 = metrics2["macro_avg"]["f1"]
    delta = f1_mac2 - f1_mac1
    comparison.append(f"Macro F1:          {f1_mac1:.4f}        {f1_mac2:.4f}  (Δ: {delta:+.4f})")

    # Weighted F1
    f1_wei1 = metrics1["weighted_avg"]["f1"]
    f1_wei2 = metrics2["weighted_avg"]["f1"]
    delta = f1_wei2 - f1_wei1
    comparison.append(f"Weighted F1:       {f1_wei1:.4f}        {f1_wei2:.4f}  (Δ: {delta:+.4f})")

    # ROC-AUC comparison (if binary)
    if metrics1.get("roc_auc") is not None and metrics2.get("roc_auc") is not None:
        roc1 = metrics1["roc_auc"]
        roc2 = metrics2["roc_auc"]
        delta = roc2 - roc1
        comparison.append(f"ROC-AUC:           {roc1:.4f}        {roc2:.4f}  (Δ: {delta:+.4f})")

    comparison.append("\n" + "=" * 70)
    return "\n".join(comparison)


# Integration helper for PyTorch models
def evaluate_pytorch_enhanced(
    model,
    loader,
    device: str = "cpu",
    output_dir: str = "outputs/reports",
    model_name: str = "radar_model",
) -> Dict[str, Any]:
    """
    Evaluate PyTorch model and compute comprehensive metrics.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model
    loader : torch.utils.data.DataLoader
        DataLoader with batches of (input, labels) or (input1, input2, ..., labels)
    device : str
        Device to run model on
    output_dir : str
        Output directory for metrics.json
    model_name : str
        Model identifier

    Returns
    -------
    Dict[str, Any]
        Comprehensive metrics dictionary
    """
    import torch

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                *inputs, labels = batch
                if len(inputs) == 1:
                    inputs = inputs[0]
            else:
                inputs, labels = batch

            # Move to device
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
            elif isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)

            labels = labels.to(device) if isinstance(labels, torch.Tensor) else labels

            # Forward pass
            if isinstance(inputs, (list, tuple)):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)

            # Get predictions and probabilities
            if isinstance(outputs, torch.Tensor):
                probs = torch.softmax(outputs, dim=1) if outputs.shape[1] > 1 else outputs
                preds = torch.argmax(outputs, dim=1)
            else:
                probs = outputs
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)

    # Convert to numpy
    predictions = np.array(all_preds)
    probabilities = np.array(all_probs)
    labels = np.array(all_labels)

    # Compute metrics
    metrics = compute_comprehensive_metrics(
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
        output_dir=output_dir,
        model_name=model_name,
    )

    return metrics


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print("ENHANCED EVALUATION PIPELINE - EXAMPLE")
    print("=" * 70 + "\n")

    # Binary classification example
    print("1. Binary Classification Example:")
    y_true_binary = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
    y_pred_binary = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
    y_probs_binary = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.85, 0.15],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.15, 0.85],
        [0.2, 0.8],
    ])

    metrics_binary = compute_comprehensive_metrics(
        predictions=y_pred_binary,
        labels=y_true_binary,
        probabilities=y_probs_binary,
        output_dir="outputs/reports",
        model_name="binary_classifier",
    )

    print(get_metrics_summary(metrics_binary))
    print(f"\n✓ Binary classification metrics saved to: {metrics_binary['_metrics_file']}")

    # Multi-class example
    print("\n" + "=" * 70)
    print("2. Multi-Class Classification Example:")
    np.random.seed(42)
    y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred_multi = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 0])
    y_probs_multi = np.random.rand(10, 3)
    y_probs_multi = y_probs_multi / y_probs_multi.sum(axis=1, keepdims=True)

    metrics_multi = compute_comprehensive_metrics(
        predictions=y_pred_multi,
        labels=y_true_multi,
        probabilities=y_probs_multi,
        output_dir="outputs/reports",
        model_name="multiclass_classifier",
    )

    print(get_metrics_summary(metrics_multi))
    print(f"\n✓ Multi-class metrics saved to: {metrics_multi['_metrics_file']}")

    print("\n" + "=" * 70)

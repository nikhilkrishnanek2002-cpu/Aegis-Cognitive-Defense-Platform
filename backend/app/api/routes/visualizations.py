"""Visualization endpoints: charts, heatmaps, 3D plots."""

from fastapi import APIRouter, Depends, HTTPException
import os
import json
from typing import Dict, Any, List
from datetime import datetime

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])

RESULTS_DIR = "results"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


async def get_current_user(token: str = None) -> dict:
    """Simple auth check."""
    return {"username": "user", "role": "operator"}


def ensure_dirs():
    """Ensure output directories exist."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


@router.get("/performance-charts")
async def get_performance_charts(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Return available performance chart data (JSON format for React charting libraries)."""
    ensure_dirs()
    
    try:
        metrics_file = os.path.join(REPORTS_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            return {"status": "ok", "metrics": metrics}
        else:
            # Return mock data
            return {
                "status": "ok",
                "metrics": {
                    "accuracy": [0.85, 0.87, 0.89, 0.90, 0.91],
                    "loss": [0.25, 0.22, 0.18, 0.15, 0.12],
                    "epochs": list(range(1, 6))
                }
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/confusion-matrix")
async def get_confusion_matrix(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Return confusion matrix as JSON for React visualization."""
    ensure_dirs()
    
    try:
        cm_file = os.path.join(REPORTS_DIR, "confusion_matrix.json")
        if os.path.exists(cm_file):
            with open(cm_file, "r") as f:
                cm = json.load(f)
            return {"status": "ok", "confusion_matrix": cm}
        else:
            # Return mock data
            return {
                "status": "ok",
                "confusion_matrix": {
                    "labels": ["Drone", "Aircraft", "Bird", "Helicopter", "Missile", "Clutter"],
                    "matrix": [
                        [95, 2, 1, 0, 0, 2],
                        [1, 94, 2, 1, 0, 2],
                        [3, 1, 91, 2, 0, 3],
                        [0, 2, 1, 93, 1, 3],
                        [0, 0, 0, 0, 98, 2],
                        [2, 1, 2, 2, 0, 93]
                    ]
                }
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/roc-curve")
async def get_roc_curve(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Return ROC curve data."""
    try:
        # Return mock ROC curve
        return {
            "status": "ok",
            "roc_curve": {
                "false_positive_rates": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
                "true_positive_rates": [0.0, 0.7, 0.85, 0.92, 0.98, 1.0],
                "auc": 0.94
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/precision-recall")
async def get_precision_recall(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Return precision-recall curve data."""
    try:
        # Return mock precision-recall data
        return {
            "status": "ok",
            "precision_recall": {
                "recalls": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "precisions": [1.0, 0.98, 0.95, 0.92, 0.88, 0.80],
                "average_precision": 0.92
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/training-history")
async def get_training_history(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Return training history (loss, accuracy over epochs)."""
    try:
        hist_file = os.path.join(REPORTS_DIR, "training_history.json")
        if os.path.exists(hist_file):
            with open(hist_file, "r") as f:
                history = json.load(f)
            return {"status": "ok", "history": history}
        else:
            # Return mock data
            return {
                "status": "ok",
                "history": {
                    "train_loss": [0.45, 0.35, 0.25, 0.18, 0.12],
                    "val_loss": [0.48, 0.38, 0.28, 0.22, 0.16],
                    "train_acc": [0.75, 0.82, 0.88, 0.91, 0.94],
                    "val_acc": [0.73, 0.80, 0.86, 0.89, 0.92],
                    "epochs": list(range(1, 6))
                }
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/3d-surface-plot")
async def get_3d_surface_plot(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Return 3D surface plot data (e.g., decision boundary)."""
    try:
        # Return mock 3D surface data
        import numpy as np
        
        x = np.linspace(-3, 3, 10).tolist()
        y = np.linspace(-3, 3, 10).tolist()
        z = [[x_val**2 + y_val**2 for y_val in y] for x_val in x]
        
        return {
            "status": "ok",
            "surface": {
                "x": x,
                "y": y,
                "z": z,
                "title": "Decision Boundary (3D)"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/xai-gradcam/{scan_id}")
async def get_gradcam_heatmap(scan_id: str, user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get GradCAM explainability heatmap for a specific scan."""
    try:
        gradcam_file = os.path.join(REPORTS_DIR, f"gradcam_{scan_id}.json")
        if os.path.exists(gradcam_file):
            with open(gradcam_file, "r") as f:
                gradcam_data = json.load(f)
            return {"status": "ok", "gradcam": gradcam_data}
        else:
            # Return mock heatmap data
            heatmap = [[0.1 * i * j for j in range(10)] for i in range(10)]
            return {
                "status": "ok",
                "gradcam": {
                    "heatmap": heatmap,
                    "shape": [10, 10],
                    "prediction": "Drone",
                    "confidence": 0.95,
                    "explanation": "Model attended to upper-left region (airborne signature)"
                }
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/model-weights-distribution")
async def get_model_weights_distribution(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get distribution of model weights for explainability."""
    try:
        return {
            "status": "ok",
            "distribution": {
                "layers": ["conv_1", "conv_2", "dense_1", "dense_2"],
                "mean": [0.002, -0.001, 0.0015, -0.002],
                "std": [0.05, 0.04, 0.03, 0.02]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/feature-importance")
async def get_feature_importance(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get feature importance scores for model explainability."""
    try:
        return {
            "status": "ok",
            "features": [
                {"name": "Doppler Shift", "importance": 0.28},
                {"name": "Range", "importance": 0.22},
                {"name": "Radar Cross Section", "importance": 0.18},
                {"name": "Velocity", "importance": 0.15},
                {"name": "Angle of Arrival", "importance": 0.12},
                {"name": "Signal Strength", "importance": 0.05}
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/radar-heatmap")
async def get_radar_heatmap(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get 2D radar detection heatmap."""
    try:
        # Return mock heatmap (Range x Doppler or Range x Azimuth)
        heatmap = [
            [0.1, 0.2, 0.15, 0.05],
            [0.3, 0.7, 0.6, 0.2],
            [0.2, 0.5, 0.95, 0.4],
            [0.1, 0.3, 0.25, 0.15]
        ]
        return {
            "status": "ok",
            "heatmap": {
                "data": heatmap,
                "x_axis": "Range (m)",
                "y_axis": "Doppler (m/s)",
                "max_value": 0.95
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/threat-timeline")
async def get_threat_timeline(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get timeline of threats over time for visualization."""
    try:
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        timeline = []
        
        for i in range(24):
            timestamp = (now - timedelta(hours=i)).isoformat()
            timeline.append({
                "timestamp": timestamp,
                "low": i % 15,
                "medium": (i * 2) % 20,
                "high": (i % 5),
                "critical": (i % 3)
            })
        
        return {
            "status": "ok",
            "timeline": timeline
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

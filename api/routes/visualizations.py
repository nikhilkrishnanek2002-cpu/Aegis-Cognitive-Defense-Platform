"""
Visualization endpoints: charts, heatmaps, 3D plots
"""
import os
import json
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime

from api.auth_utils import get_current_user
from src.logger import log_event

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])

RESULTS_DIR = "results"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def ensure_dirs():
    """Ensure output directories exist."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


@router.get("/performance-charts")
async def get_performance_charts(user: dict = Depends(get_current_user)):
    """Return available performance chart data (JSON format for React charting libraries)."""
    ensure_dirs()
    
    try:
        metrics_file = os.path.join(REPORTS_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            return {"status": "ok", "metrics": metrics}
        else:
            return {"status": "no_metrics", "message": "No metrics available yet"}
    except Exception as e:
        log_event(f"Error loading performance charts: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confusion-matrix")
async def get_confusion_matrix(user: dict = Depends(get_current_user)):
    """Return confusion matrix as JSON for React visualization."""
    ensure_dirs()
    
    try:
        cm_file = os.path.join(REPORTS_DIR, "confusion_matrix.json")
        if os.path.exists(cm_file):
            with open(cm_file, "r") as f:
                cm_data = json.load(f)
            return {"status": "ok", "data": cm_data}
        else:
            return {"status": "no_data", "message": "Confusion matrix not available"}
    except Exception as e:
        log_event(f"Error loading confusion matrix: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roc-curve")
async def get_roc_curve(user: dict = Depends(get_current_user)):
    """Return ROC curve data as JSON for React visualization."""
    ensure_dirs()
    
    try:
        roc_file = os.path.join(REPORTS_DIR, "roc_curve.json")
        if os.path.exists(roc_file):
            with open(roc_file, "r") as f:
                roc_data = json.load(f)
            return {"status": "ok", "data": roc_data}
        else:
            return {"status": "no_data", "message": "ROC curve not available"}
    except Exception as e:
        log_event(f"Error loading ROC curve: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/precision-recall")
async def get_precision_recall(user: dict = Depends(get_current_user)):
    """Return precision-recall curve data."""
    ensure_dirs()
    
    try:
        pr_file = os.path.join(REPORTS_DIR, "precision_recall.json")
        if os.path.exists(pr_file):
            with open(pr_file, "r") as f:
                pr_data = json.load(f)
            return {"status": "ok", "data": pr_data}
        else:
            return {"status": "no_data", "message": "Precision-recall curve not available"}
    except Exception as e:
        log_event(f"Error loading precision-recall: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-history")
async def get_training_history(user: dict = Depends(get_current_user)):
    """Return training history for loss/accuracy curves."""
    ensure_dirs()
    
    try:
        hist_file = os.path.join(REPORTS_DIR, "training_history.json")
        if os.path.exists(hist_file):
            with open(hist_file, "r") as f:
                hist_data = json.load(f)
            return {"status": "ok", "data": hist_data}
        else:
            return {"status": "no_data", "message": "Training history not available"}
    except Exception as e:
        log_event(f"Error loading training history: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/3d-surface-plot")
async def get_3d_surface_data(user: dict = Depends(get_current_user)):
    """Return 3D surface plot data (e.g., Range-Doppler map elevation)."""
    ensure_dirs()
    
    try:
        data_3d_file = os.path.join(REPORTS_DIR, "surface_3d.json")
        if os.path.exists(data_3d_file):
            with open(data_3d_file, "r") as f:
                surface_data = json.load(f)
            return {"status": "ok", "data": surface_data}
        else:
            return {"status": "no_data", "message": "3D surface data not available"}
    except Exception as e:
        log_event(f"Error loading 3D data: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/network-graph")
async def get_network_graph(user: dict = Depends(get_current_user)):
    """Return network/graph data for decision tree or model architecture."""
    ensure_dirs()
    
    try:
        graph_file = os.path.join(REPORTS_DIR, "network_graph.json")
        if os.path.exists(graph_file):
            with open(graph_file, "r") as f:
                graph_data = json.load(f)
            return {"status": "ok", "data": graph_data}
        else:
            return {"status": "no_data", "message": "Network graph not available"}
    except Exception as e:
        log_event(f"Error loading network graph: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/xai-gradcam/{scan_id}")
async def get_xai_gradcam(scan_id: str, user: dict = Depends(get_current_user)):
    """Return Grad-CAM heatmap data for a specific scan."""
    ensure_dirs()
    
    try:
        xai_file = os.path.join(REPORTS_DIR, f"gradcam_{scan_id}.json")
        if os.path.exists(xai_file):
            with open(xai_file, "r") as f:
                xai_data = json.load(f)
            return {"status": "ok", "data": xai_data}
        else:
            return {"status": "no_data", "message": f"Grad-CAM data not found for scan {scan_id}"}
    except Exception as e:
        log_event(f"Error loading Grad-CAM: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/xai-gradcam-image/{scan_id}")
async def get_xai_gradcam_image(scan_id: str, user: dict = Depends(get_current_user)):
    """Return Grad-CAM heatmap as PNG image."""
    try:
        xai_img = os.path.join(REPORTS_DIR, f"gradcam_{scan_id}.png")
        if os.path.exists(xai_img):
            return FileResponse(xai_img, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail=f"Image not found for scan {scan_id}")
    except Exception as e:
        log_event(f"Error serving Grad-CAM image: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-training-charts")
async def generate_training_charts(user: dict = Depends(get_current_user)):
    """Trigger generation of training performance charts from current metrics."""
    ensure_dirs()
    
    try:
        from src.reporting import (
            plot_confusion_matrix, plot_roc_curve, 
            plot_precision_recall, plot_training_history
        )
        from src.evaluation_enhanced import EvaluationMetrics
        
        metrics_file = os.path.join(REPORTS_DIR, "metrics.json")
        if not os.path.exists(metrics_file):
            return {"status": "error", "message": "No metrics available"}
        
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        # Generate charts if data is available
        y_true = metrics.get("y_true", [])
        y_pred = metrics.get("y_pred", [])
        
        if y_true and y_pred:
            cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
            plot_confusion_matrix(y_true, y_pred, cm_path)
            log_event(f"Generated confusion matrix: {cm_path}", level="info")
        
        log_event("Training charts generated", level="info")
        return {"status": "ok", "message": "Charts generated successfully"}
    
    except Exception as e:
        log_event(f"Error generating charts: {e}", level="error")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    ensure_dirs()
    print(f"✓ Visualization routes ready. Output dir: {REPORTS_DIR}")

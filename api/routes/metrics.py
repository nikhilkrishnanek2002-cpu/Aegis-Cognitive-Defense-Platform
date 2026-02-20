"""
Research metrics: classification report JSON and chart images.
"""
import json
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from api.auth_utils import get_current_user

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

METRICS_JSON_PATH = os.path.join("outputs", "reports", "metrics.json")
METRIC_IMAGE_PATHS = {
    "confusion_matrix": os.path.join("results", "reports", "confusion_matrix.png"),
    "roc_curve": os.path.join("results", "reports", "roc_curve.png"),
    "precision_recall": os.path.join("results", "reports", "precision_recall.png"),
    "training_curves": os.path.join("results", "reports", "training_history.png"),
}


@router.get("/report")
async def get_report(user: dict = Depends(get_current_user)):
    if not os.path.exists(METRICS_JSON_PATH):
        raise HTTPException(status_code=404, detail="Metrics report not found")
    try:
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Metrics file corrupted: {exc}")


@router.get("/images/{name}")
async def get_image(name: str, user: dict = Depends(get_current_user)):
    path = METRIC_IMAGE_PATHS.get(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Unknown image: {name}. Valid: {list(METRIC_IMAGE_PATHS)}")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
    return FileResponse(path, media_type="image/png")

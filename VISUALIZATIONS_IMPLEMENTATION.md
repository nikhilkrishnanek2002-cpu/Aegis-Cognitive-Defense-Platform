# Performance Charts & 3D Visualizations - Implementation Summary

## ✅ What Was Added

Your Aegis platform now has complete **Performance Charts**, **Explainable AI (Grad-CAM)**, and **3D Visualizations** integrated into the FastAPI/React frontend.

---

## 📊 Components Created

### 1. **API Endpoints** (`api/routes/visualizations.py`)
New FastAPI endpoints for generating and serving visualizations:
- `/api/visualizations/performance-charts` - Performance metric data
- `/api/visualizations/confusion-matrix` - Confusion matrix JSON
- `/api/visualizations/roc-curve` - ROC curve data
- `/api/visualizations/precision-recall` - Precision-recall curve data
- `/api/visualizations/training-history` - Training loss/accuracy history
- `/api/visualizations/3d-surface-plot` - 3D surface plot data
- `/api/visualizations/network-graph` - Network/decision tree graphs
- `/api/visualizations/xai-gradcam/{scan_id}` - Grad-CAM heatmap JSON
- `/api/visualizations/xai-gradcam-image/{scan_id}` - Grad-CAM heatmap PNG

### 2. **Enhanced Radar Scan Endpoint**
The `/api/radar/scan` endpoint now:
- ✅ Generates **Grad-CAM heatmaps** for each detection
- ✅ Saves heatmaps as both PNG and JSON
- ✅ Returns XAI data in the response (`xai` field)
- ✅ Generates unique scan IDs for tracking

### 3. **React Components**

#### `PerformanceChartsComponent.tsx`
Interactive Plotly charts for:
- Confusion Matrix (heatmap)
- ROC Curve (with AUC score)
- Precision-Recall Curve
- Training Progress (loss over epochs)

#### `Visualization3DComponent.tsx`
3D and 2D interactive visualizations:
- **3D Range-Doppler Surface Plot** - 3D elevation of RD map
- **3D Detection Scatter** - Point cloud of detections in 3D space
- **3D Spectrogram Surface** - 3D elevation of frequency spectrum
- **2D Range-Doppler Heatmap** - Top-down view with color intensity

#### **Updated XAITab.tsx**
- Displays Grad-CAM heatmaps from scan results
- Shows target class and confidence level
- Button to generate new Grad-CAM visualizations
- Interactive Plotly heatmap viewer
- PNG image overlay display

#### **Updated MetricsTab.tsx**
- **Tab Navigation**: Summary | Performance Charts | 3D Visualizations
- **Summary Tab**: Model metadata and per-class metrics
- **Charts Tab**: Dynamic performance charts using Plotly
- **3D Tab**: Full 3D visualizations of radar data

---

## 🚀 How to Use

### 1. **Run a Radar Scan**
```bash
# Navigate to Analytics tab and click "Run Scan"
# or via curl:
curl -X POST http://localhost:8000/api/radar/scan \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"target": "drone", "distance": 200, "gain_db": 15}'
```

The response now includes XAI data:
```json
{
  "scan_id": "a1b2c3d4",
  "detected": "Drone",
  "confidence": 0.95,
  "xai": {
    "scan_id": "a1b2c3d4",
    "heatmap": [[]],  // Grad-CAM matrix
    "heatmap_shape": [128, 128],
    "target_class": "Drone",
    "confidence": 0.95,
    "image_path": "/api/visualizations/xai-gradcam-image/a1b2c3d4"
  },
  "rd_map": [[]],    // Range-Doppler map
  "spec": [[]],      // Spectrogram
  ...
}
```

### 2. **View Performance Charts**
Navigate to **Metrics Tab** → **Performance Charts** to see:
- Confusion matrix heatmap
- ROC curve with AUC
- Precision-recall curves
- Training loss curves

### 3. **View 3D Visualizations**
Navigate to **Metrics Tab** → **3D Visualizations** to see:
- 3D surface plot of Range-Doppler map
- 3D scatter plot of detections
- 3D surface plot of spectrogram
- 2D heatmap for reference

### 4. **View Grad-CAM Explanations**
Navigate to **Explainable AI Tab** to see:
- Interactive Grad-CAM heatmap (shows which regions influenced classification)
- Confidence and scan ID
- PNG overlay image
- "Generate Grad-CAM" button to create new visualizations

---

## 📁 Files Modified

### Backend
- `api/main.py` - Added visualizations router
- `api/routes/radar.py` - Enhanced with Grad-CAM generation
- `api/routes/visualizations.py` - **NEW** - All visualization endpoints

### Frontend
- `frontend/src/api/client.ts` - Added API_BASE export
- `frontend/src/components/PerformanceChartsComponent.tsx` - **NEW**
- `frontend/src/components/Visualization3DComponent.tsx` - **NEW**
- `frontend/src/components/tabs/XAITab.tsx` - Updated with real Grad-CAM
- `frontend/src/components/tabs/MetricsTab.tsx` - Updated with tabs

---

## 🔧 Technical Details

### Grad-CAM Generation
- Uses `AIReliabilityHardener` and `GradCAMExplainer` from `src/ai_hardening.py`
- Generates on every scan for the highest-confidence detection
- Saved to: `results/reports/gradcam_{scan_id}.{json|png}`
- Normalized to 0-255 for PNG display

### Data Flow
```
User runs scan
    ↓
/api/radar/scan endpoint
    ↓
Generate Grad-CAM heatmap
    ↓
Save to results/reports/
    ↓
Return scan with xai field
    ↓
Frontend displays in XAI tab
    ↓
User clicks on Grad-CAM to interact with visualization
```

### 3D Visualization Libraries
- **Plotly.js** - Interactive 3D surface plots, scatter plots, heatmaps
- Supports rotation, zoom, pan on 3D plots
- Automatic color scaling based on data

---

## 📦 Dependencies

All required packages already installed:
- `react-plotly.js` - React wrapper for Plotly
- `plotly.js-dist-min` - Plotly runtime

Python backend dependencies (already in requirements.txt):
- `plotly` - Python Plotly
- `matplotlib` - For PNG chart generation (fallback)
- `scikit-learn` - Metrics and confusion matrices

---

## 🎨 Visualization Features

### Performance Charts
✅ Confusion Matrix - Shows classification accuracy per class
✅ ROC Curve - Receiver Operating Characteristic with AUC
✅ Precision-Recall Curve - F1 trade-off visualization
✅ Training History - Loss and accuracy over epochs

### 3D Visualizations
✅ Range-Doppler 3D Surface - Elevation map of range vs doppler
✅ Detection Scatter 3D - Point cloud with power levels
✅ Spectrogram 3D - Elevation map of frequency spectrum
✅ 2D Range-Doppler - Traditional heatmap top-down view

### Explainability (XAI)
✅ Grad-CAM Heatmap - Shows regions important for classification
✅ Confidence Display - Model confidence in the decision
✅ PNG + JSON - Both image and data formats

---

## 🔍 Troubleshooting

### No Grad-CAM appearing?
1. Ensure you've run at least one scan (`/api/radar/scan`)
2. Check that the model file exists: `radar_model_pytorch.pt`
3. Look for errors in browser console (F12 → Console tab)
4. Check backend logs for XAI generation errors

### Charts not loading?
1. Verify metrics data exists: `results/reports/metrics.json`
2. Check that training has completed to generate metrics
3. Ensure Plotly is loaded: `npm list react-plotly.js` in frontend

### 3D plots very slow?
1. Reduce data size if matrices are > 256x256
2. Browser rendering performance depends on GPU
3. Chrome typically faster than Firefox for WebGL

---

## 🎯 Next Steps

To fully leverage these features:

1. **Train the model** to generate metrics data
2. **Run scan operations** to generate Grad-CAM heatmaps
3. **View results** in the UI tabs

Example workflow:
```bash
# 1. Train model (generates metrics.json)
python experiment_runner.py --mode train

# 2. Run inference - generates Grad-CAM
curl -X POST http://localhost:8000/api/radar/scan \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"target": "drone", "distance": 200, "gain_db": 15}'

# 3. View in UI
# - Navigate to Metrics tab → Performance Charts (see training results)
# - Navigate to XAI tab (see Grad-CAM for the scan)
# - Navigate back to Metrics → 3D Visualizations (see live radar data)
```

---

✅ **Implementation Complete!** Your platform now has full visualization support.

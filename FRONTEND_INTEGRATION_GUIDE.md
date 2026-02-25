# Frontend Integration Guide: Multi-Resolution XAI Heatmaps

**API Change Summary**: Backend now returns multi-resolution heatmap data with system health metrics  
**Backward Compatible**: Yes, existing code still works (uses `heatmap` field)  
**Enhancement**: New fields provide progressive loading and system visibility

---

## 📦 New Response Structure

### Complete XAI Response Format
```javascript
{
  scan_id: "abc12345",                    // Unique scan identifier
  timestamp: 1704067200.12,              // Unix timestamp
  
  // === MULTI-RESOLUTION HEATMAPS ===
  heatmap: [                              // 128×128 standard heatmap
    [0.0, 0.2, ..., 0.8],
    [0.1, 0.5, ..., 0.9],
    ...
  ],
  heatmap_thumbnail: [                    // 32×32 quick preview
    [0.0, 0.2, ..., 0.8],
    ...
  ],
  heatmap_detail: [                       // 256×256 detailed analysis (or null)
    [0.0, 0.2, ..., 0.8],
    ...
  ] || null,
  
  // === SHAPE INFORMATION ===
  heatmap_shape: [128, 128],              // Standard heatmap dimensions
  heatmap_shape_detail: [256, 256],       // Detail heatmap dimensions (or null)
  
  // === DETECTION METADATA ===
  target_class: "Drone",                  // Detected target type
  confidence: 0.9542,                     // Confidence score
  image_path: "/api/visualizations/xai-gradcam-image/abc12345",
  
  // === ADAPTIVE INFO ===
  generation_mode: "real|synthetic|emergency",  // How heatmap was generated
  adaptive_resolution: 128,               // Actual resolution used
  multi_resolution_support: true,         // New feature support flag
  
  // === SYSTEM HEALTH ===
  system_health: {
    cpu_percent: 45.2,                   // Average CPU usage
    memory_percent: 62.1,                // Average memory usage
    avg_latency_ms: 85.3,                // Average operation latency
    is_healthy: true                     // System health status
  },
  
  // === ERROR RECOVERY ===
  error_fallback: false,                 // True if emergency fallback engaged
}
```

---

## 🎨 Frontend Display Strategies

### Strategy 1: Progressive Loading (Recommended)
Display thumbnail immediately, upgrade to standard quality when available:

```javascript
// React component example
function XAIHeatmapDisplay({ xaiData }) {
  const [displayMode, setDisplayMode] = useState('thumbnail');
  
  useEffect(() => {
    if (!xaiData) return;
    
    // Show thumbnail immediately
    setDisplayMode('thumbnail');
    
    // Upgrade to standard after short delay
    const timer = setTimeout(() => {
      setDisplayMode('standard');
    }, 100);
    
    return () => clearTimeout(timer);
  }, [xaiData]);
  
  const getData = () => {
    if (displayMode === 'standard') return xaiData.heatmap;
    if (displayMode === 'detail') return xaiData.heatmap_detail;
    return xaiData.heatmap_thumbnail;
  };
  
  const getSize = () => {
    if (displayMode === 'standard') return xaiData.heatmap_shape;
    if (displayMode === 'detail') return xaiData.heatmap_shape_detail;
    return [32, 32];
  };
  
  return (
    <div className="heatmap-container">
      <HeatmapRenderer 
        data={getData()} 
        size={getSize()}
        label={displayMode.toUpperCase()}
      />
      
      {xaiData.heatmap_detail && displayMode !== 'detail' && (
        <button onClick={() => setDisplayMode('detail')}>
          View Details (256×256)
        </button>
      )}
    </div>
  );
}
```

### Strategy 2: Toggle Between Resolutions
Let operators choose detail level:

```javascript
function AdvancedHeatmapView({ xaiData }) {
  const [resolution, setResolution] = useState('standard');
  
  const getHeatmapData = () => {
    return {
      thumbnail: { data: xaiData.heatmap_thumbnail, size: [32, 32] },
      standard: { data: xaiData.heatmap, size: xaiData.heatmap_shape },
      detail: { data: xaiData.heatmap_detail, size: xaiData.heatmap_shape_detail }
    }[resolution];
  };
  
  return (
    <div>
      <div className="resolution-selector">
        <button 
          className={resolution === 'thumbnail' ? 'active' : ''}
          onClick={() => setResolution('thumbnail')}
        >
          Quick (32×32)
        </button>
        <button 
          className={resolution === 'standard' ? 'active' : ''}
          onClick={() => setResolution('standard')}
        >
          Standard (128×128) ← Default
        </button>
        {xaiData.heatmap_detail && (
          <button 
            className={resolution === 'detail' ? 'active' : ''}
            onClick={() => setResolution('detail')}
            disabled={!xaiData.system_health.is_healthy}
            title={!xaiData.system_health.is_healthy ? 'Detail unavailable under load' : ''}
          >
            Details (256×256)
          </button>
        )}
      </div>
      
      <HeatmapDisplay {...getHeatmapData()} />
    </div>
  );
}
```

### Strategy 3: System-Aware Display
Show different details based on system health:

```javascript
function SmartHeatmapDisplay({ xaiData }) {
  const health = xaiData.system_health;
  
  // Determine recommended resolution
  const getRecommendedResolution = () => {
    if (!health.is_healthy) return 'thumbnail'; // Fast mode
    if (health.cpu_percent > 70 || health.memory_percent > 70) return 'standard';
    return 'detail'; // Show detail if system healthy and available
  };
  
  return (
    <div className="xai-display-container">
      <div className="heatmap-section">
        {/* Main heatmap */}
        <HeatmapRenderer 
          data={xaiData.heatmap} 
          size={xaiData.heatmap_shape}
          title={`Heatmap @ ${xaiData.adaptive_resolution}×${xaiData.adaptive_resolution}`}
        />
        
        {/* System health badge */}
        <div className={`health-badge ${health.is_healthy ? 'healthy' : 'stressed'}`}>
          <div>CPU: {health.cpu_percent.toFixed(1)}%</div>
          <div>Mem: {health.memory_percent.toFixed(1)}%</div>
          <div>Latency: {health.avg_latency_ms.toFixed(1)}ms</div>
        </div>
      </div>
      
      {/* Detail heatmap (if available and system healthy) */}
      {xaiData.heatmap_detail && health.is_healthy && (
        <div className="detail-section">
          <HeatmapRenderer 
            data={xaiData.heatmap_detail}
            size={xaiData.heatmap_shape_detail}
            title="Detailed Analysis (256×256)"
          />
        </div>
      )}
      
      {/* Loading state or warnings */}
      {!health.is_healthy && (
        <div className="warning">⚠️ System under load - detail heatmap not available</div>
      )}
    </div>
  );
}
```

---

## 🔄 Heatmap Rendering

### Canvas-Based Renderer
Convert heatmap array to visual display:

```javascript
function HeatmapRenderer({ data, size, title = '' }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!canvasRef.current || !data) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const [width, height] = size;
    
    canvas.width = width;
    canvas.height = height;
    
    // Create image data
    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data;
    
    // Flatten and normalize data
    const flat = data.flat();
    
    for (let i = 0; i < flat.length; i++) {
      const intensity = Math.round(flat[i] * 255);
      const pixelIndex = i * 4;
      
      // Apply color map (red-yellow gradient)
      if (intensity < 128) {
        pixels[pixelIndex] = intensity * 2;      // Red channel
        pixels[pixelIndex + 1] = 0;               // Green channel
        pixels[pixelIndex + 2] = 0;               // Blue channel
      } else {
        pixels[pixelIndex] = 255;                 // Red channel
        pixels[pixelIndex + 1] = (intensity - 128) * 2;  // Green channel
        pixels[pixelIndex + 2] = 0;               // Blue channel
      }
      
      pixels[pixelIndex + 3] = 200;               // Alpha (slightly transparent)
    }
    
    ctx.putImageData(imageData, 0, 0);
  }, [data, size]);
  
  return (
    <div className="heatmap-wrapper">
      {title && <div className="heatmap-title">{title}</div>}
      <canvas 
        ref={canvasRef}
        className="heatmap-canvas"
        style={{
          border: '1px solid #ccc',
          imageRendering: 'pixelated',  // Preserve pixel boundaries
          width: '256px',
          height: '256px'
        }}
      />
    </div>
  );
}
```

---

## 📊 System Health Monitoring

### Display System Status
Show operators the system load and heatmap quality:

```javascript
function SystemHealthMonitor({ health }) {
  const getCpuColor = (pct) => {
    if (pct < 50) return 'green';
    if (pct < 70) return 'yellow';
    if (pct < 85) return 'orange';
    return 'red';
  };
  
  const getMemColor = (pct) => {
    if (pct < 60) return 'green';
    if (pct < 75) return 'yellow';
    if (pct < 85) return 'orange';
    return 'red';
  };
  
  return (
    <div className="health-monitor">
      <div className="metric">
        <label>CPU Usage</label>
        <div className="bar">
          <div 
            className="fill"
            style={{
              width: `${health.cpu_percent}%`,
              backgroundColor: getCpuColor(health.cpu_percent)
            }}
          />
        </div>
        <span>{health.cpu_percent.toFixed(1)}%</span>
      </div>
      
      <div className="metric">
        <label>Memory Usage</label>
        <div className="bar">
          <div 
            className="fill"
            style={{
              width: `${health.memory_percent}%`,
              backgroundColor: getMemColor(health.memory_percent)
            }}
          />
        </div>
        <span>{health.memory_percent.toFixed(1)}%</span>
      </div>
      
      <div className="metric">
        <label>Avg Latency</label>
        <span className={health.avg_latency_ms > 100 ? 'warning' : ''}>
          {health.avg_latency_ms.toFixed(1)}ms
        </span>
      </div>
      
      <div className="status">
        <label>System Status</label>
        <span className={health.is_healthy ? 'healthy' : 'stressed'}>
          {health.is_healthy ? '✅ Healthy' : '⚠️ Under Load'}
        </span>
      </div>
    </div>
  );
}
```

---

## 🔄 Backward Compatibility

### Old Code Still Works
Existing code using `heatmap` field continues to function:

```javascript
// OLD CODE (still works)
function OldXAIDisplay({ xaiData }) {
  return (
    <HeatmapRenderer 
      data={xaiData.heatmap}  // 128×128 standard
      size={xaiData.heatmap_shape}
    />
  );
}
```

### Enhanced Code (Recommended)
Use new fields for better UX:

```javascript
// NEW CODE (enhanced)
function EnhancedXAIDisplay({ xaiData }) {
  // Use multi-resolution support
  const [showDetail, setShowDetail] = useState(false);
  
  return (
    <>
      <HeatmapRenderer 
        data={showDetail ? xaiData.heatmap_detail : xaiData.heatmap}
        size={showDetail ? xaiData.heatmap_shape_detail : xaiData.heatmap_shape}
      />
      
      {xaiData.heatmap_detail && (
        <button onClick={() => setShowDetail(!showDetail)}>
          {showDetail ? 'Show Standard' : 'Show Details'}
        </button>
      )}
      
      <SystemHealthMonitor health={xaiData.system_health} />
    </>
  );
}
```

---

## 📋 Field Reference

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `heatmap` | 2D Array | 128×128 | Standard display heatmap (always present) |
| `heatmap_thumbnail` | 2D Array | 32×32 | Quick preview (always present) |
| `heatmap_detail` | 2D Array | 256×256 | Detailed analysis (optional, may be null) |
| `heatmap_shape` | Array | 2 | Dimensions [height, width] of standard |
| `heatmap_shape_detail` | Array | 2 | Dimensions of detail (or null) |
| `target_class` | String | - | Detected class (e.g., "Drone") |
| `confidence` | Float | - | Confidence 0.0-1.0 |
| `generation_mode` | String | - | "real", "synthetic", or "emergency" |
| `adaptive_resolution` | Int | - | Resolution used (32, 64, 96, 128, etc) |
| `multi_resolution_support` | Boolean | - | Always true (new feature flag) |
| `system_health.cpu_percent` | Float | - | CPU usage 0-100 |
| `system_health.memory_percent` | Float | - | Memory usage 0-100 |
| `system_health.avg_latency_ms` | Float | - | Average latency in milliseconds |
| `system_health.is_healthy` | Boolean | - | True if CPU<85% and Mem<85% |

---

## 🎯 Best Practices

### DO ✅
- Use thumbnail for quick preview
- Show detail only if `system_health.is_healthy`
- Display system health alongside heatmap
- Handle null `heatmap_detail` gracefully
- Progress update to detail when available

### DON'T ❌
- Always request 256×256 (may not be available under load)
- Ignore `multi_resolution_support` flag
- Force detail rendering when system stressed
- Assume `heatmap_detail` is always present
- Forget to handle emergency fallback mode

---

## 🚀 Implementation Examples

### React Hooks Example
```javascript
const [xaiData, setXaiData] = useState(null);
const [loading, setLoading] = useState(false);

useEffect(() => {
  setLoading(true);
  fetch('/api/radar/scan')
    .then(r => r.json())
    .then(data => {
      // New multi-resolution response
      setXaiData(data);
      
      // Log system state
      console.log(
        `System: CPU ${data.system_health.cpu_percent}% / ` +
        `Mem ${data.system_health.memory_percent}% / ` +
        `Mode: ${data.generation_mode}`
      );
    })
    .finally(() => setLoading(false));
}, []);

// Display with progressive loading
return (
  <>
    {loading && <div>Loading...</div>}
    {xaiData && (
      <>
        <HeatmapRenderer data={xaiData.heatmap} />
        <SystemHealthMonitor health={xaiData.system_health} />
      </>
    )}
  </>
);
```

---

## 📞 Troubleshooting

**Q: Why is `heatmap_detail` null?**  
A: System is under load (CPU>70% or Memory>70%). Detail generation suspended for performance.

**Q: What does `generation_mode: 'emergency'` mean?**  
A: Both real and synthetic generation failed. Using minimal fallback (32×32) to ensure system stability.

**Q: Is the old `heatmap` field still there?**  
A: Yes, fully backward compatible. Still contains 128×128 standard heatmap.

**Q: How do I know if detail is available?**  
A: Check three conditions: (1) `heatmap_detail` is not null, (2) `heatmap_shape_detail` is [256, 256], (3) `system_health.is_healthy` is true.

**Q: What's the payload size difference?**  
A: Standard 128×128 ≈ 64 KB. Adding 32×32 thumbnail adds ~1 KB. Detail 256×256 adds ~256 KB (only when available).

---

## 🎓 Summary

**Key Changes:**
- ✅ Multi-resolution heatmap data in response
- ✅ System health metrics for operator awareness
- ✅ Adaptive resolution information
- ✅ Progressive loading support
- ✅ Backward compatible

**Recommended Implementation:**
1. Display thumbnail immediately for responsiveness
2. Upgrade to standard after brief delay
3. Show detail only if system healthy
4. Display health metrics alongside heatmap
5. Update UI based on generation mode

**No breaking changes - implement at your pace!** 🚀

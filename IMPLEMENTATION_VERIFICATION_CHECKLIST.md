# Implementation Verification Checklist

**Purpose**: Step-by-step validation of all safety improvements  
**Status**: Ready for testing  
**Duration**: 15-20 minutes for complete verification  
**Last Updated**: Current Session

---

## ✅ Phase 1: Syntax & Import Verification

### 1.1 File Integrity
- [ ] File exists: `api/routes/radar.py`
- [ ] File size: ~619 lines (verify with `wc -l`)
- [ ] No corrupted characters: Run through syntax checker
- [ ] Command: `python -m py_compile api/routes/radar.py`

### 1.2 Import Verification
- [ ] psutil imported: `import psutil`
- [ ] hashlib imported: `import hashlib`
- [ ] deque imported: `from collections import deque`
- [ ] Lock imported: `from threading import Lock`
- [ ] All imports at lines 1-25

**Test Command:**
```bash
python -c "from api.routes.radar import _health_monitor, _heatmap_cache; print('✓ Imports OK')"
```

### 1.3 Class Initialization
- [ ] SystemHealthMonitor class defined
- [ ] HeatmapCache class defined
- [ ] Both instantiated: `_health_monitor`, `_heatmap_cache`
- [ ] No initialization errors

**Test Command:**
```bash
python -c "from api.routes.radar import _health_monitor; print(f'Monitor: {type(_health_monitor).__name__}')"
```

---

## ✅ Phase 2: Infrastructure Functionality

### 2.1 SystemHealthMonitor
- [ ] `update()` method completes without error
- [ ] `get_avg_cpu()` returns float 0-100
- [ ] `get_avg_memory()` returns float 0-100
- [ ] `add_latency()` accepts float values
- [ ] `get_avg_latency()` returns float ≥ 0
- [ ] `is_system_healthy()` returns boolean
- [ ] `is_system_under_load()` returns boolean
- [ ] Thread safety: Lock is used in all methods

**Test Code:**
```python
from api.routes.radar import _health_monitor

# Update health
_health_monitor.update()

# Get metrics
cpu = _health_monitor.get_avg_cpu()
mem = _health_monitor.get_avg_memory()
lat = _health_monitor.get_avg_latency()

# Check health
healthy = _health_monitor.is_system_healthy()
loaded = _health_monitor.is_system_under_load()

print(f"CPU: {cpu}% | Mem: {mem}% | Latency: {lat}ms")
print(f"Healthy: {healthy} | Loaded: {loaded}")

# Record latency
for i in range(10):
    _health_monitor.add_latency(50.0 + i*5)

avg_lat = _health_monitor.get_avg_latency()
print(f"Average latency: {avg_lat}ms")
```

### 2.2 HeatmapCache
- [ ] `_compute_key()` generates MD5 hash string
- [ ] `get()` returns cached data or None
- [ ] `put()` stores data and returns True
- [ ] `clear_expired()` removes old entries
- [ ] Cache respects max_items limit (500)
- [ ] Cache respects max_memory limit (200 MB)
- [ ] LRU eviction works correctly
- [ ] Thread safety: Lock is used in all methods

**Test Code:**
```python
from api.routes.radar import _heatmap_cache
import numpy as np

# Generate test data
test_cam = np.random.rand(128, 128)
test_data = {
    "heatmap": test_cam.tolist(),
    "scan_id": "test123",
    "generation_mode": "synthetic"
}

# Test cache operations
key = _heatmap_cache._compute_key("det1", "v2.0", 128)
print(f"Cache key: {key[:16]}... (first 16 chars)")

# Store
success = _heatmap_cache.put(key, test_data)
print(f"Put success: {success}")

# Retrieve
cached = _heatmap_cache.get(key)
print(f"Cache hit: {cached is not None}")

# Verify data
if cached:
    print(f"Cached scan_id: {cached['scan_id']}")
    print(f"Cached mode: {cached['generation_mode']}")
```

---

## ✅ Phase 3: Adaptive Resolution

### 3.1 Function Exists
- [ ] `_calculate_adaptive_heatmap_size()` function defined
- [ ] Function signature: `(threat_level: str, ew_active: bool = False) -> int`
- [ ] Returns valid integer size

### 3.2 Resolution Mapping
Test each threat level:

- [ ] Green threat + healthy system: Returns 128
- [ ] Green threat + under load: Returns 96
- [ ] Yellow threat: Returns 128
- [ ] Red threat + healthy system: Returns 256
- [ ] Red threat + under load: Returns 128
- [ ] Unknown threat: Returns 128 (default)

**Test Code:**
```python
from api.routes.radar import _calculate_adaptive_heatmap_size, _health_monitor

# Test different scenarios
test_cases = [
    ('green', False, 'Normal green'),
    ('yellow', False, 'Normal yellow'),
    ('red', False, 'Normal red'),
    ('red', True, 'Red with EW active'),
]

for threat, ew_active, label in test_cases:
    size = _calculate_adaptive_heatmap_size(threat, ew_active)
    print(f"{label}: {size}×{size} ({size*size} pixels)")

# Valid sizes
valid_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
print(f"\nAll sizes within valid range: {all([size in valid_sizes for t, e, l in test_cases for size in [_calculate_adaptive_heatmap_size(t, e)]])}")
```

### 3.3 Load-Based Scaling
- [ ] Under normal load: Returns base size
- [ ] Under high load (CPU>70%): Returns smaller size
- [ ] Critical load (CPU>85%): Returns 32 (minimum)
- [ ] Memory check also triggers downscaling

---

## ✅ Phase 4: Multi-Resolution Generation

### 4.1 Function Exists
- [ ] `_generate_multiresolution_heatmaps()` function defined
- [ ] Function signature: `(base_cam: np.ndarray, compute_detail: bool = False) -> Dict`
- [ ] Returns dictionary with correct keys

### 4.2 Output Format
- [ ] Returns dict with keys: 'thumbnail', 'standard', 'detail'
- [ ] 'thumbnail' is always present (32×32)
- [ ] 'standard' is always present (128×128)
- [ ] 'detail' is list or None
- [ ] All values are lists (not arrays)

**Test Code:**
```python
from api.routes.radar import _generate_multiresolution_heatmaps
import numpy as np

# Create test heatmap
test_cam = np.random.rand(128, 128)

# Generate multi-resolution
result = _generate_multiresolution_heatmaps(test_cam, compute_detail=False)

print(f"Keys: {list(result.keys())}")
print(f"Thumbnail shape: {len(result['thumbnail'])}×{len(result['thumbnail'][0])if result['thumbnail'] else 0}")
print(f"Standard shape: {len(result['standard'])}×{len(result['standard'][0])if result['standard'] else 0}")
print(f"Detail available: {result['detail'] is not None}")

# Check when detail available
result_detail = _generate_multiresolution_heatmaps(test_cam, compute_detail=True)
print(f"Detail with compute_detail=True: {result_detail['detail'] is not None}")
```

### 4.3 Size Verification
- [ ] Thumbnail: 32×32 (1 KB)
- [ ] Standard: 128×128 (64 KB)
- [ ] Detail: 256×256 or None (256 KB if present)
- [ ] All are valid NumPy arrays converted to lists

---

## ✅ Phase 5: Synthetic Grad-CAM

### 5.1 Function Behavior
- [ ] `_generate_synthetic_gradcam(size=128)` works with different sizes
- [ ] Returns NumPy array of shape (size, size)
- [ ] Values in range [0.0, 1.0]
- [ ] Generates smooth Gaussian-like distribution

**Test Code:**
```python
from api.routes.radar import _generate_synthetic_gradcam
import numpy as np

# Test different sizes
for size in [32, 64, 96, 128, 256]:
    cam = _generate_synthetic_gradcam(size=size)
    print(f"Size {size}: shape={cam.shape}, min={cam.min():.3f}, max={cam.max():.3f}, mean={cam.mean():.3f}")
    assert cam.shape == (size, size), f"Wrong shape for size {size}"
    assert 0 <= cam.min() and cam.max() <= 1, f"Values out of range for size {size}"
```

### 5.2 Fallback Reliability
- [ ] Works when no errors occur
- [ ] Always returns valid data (never None)
- [ ] Handles edge cases (size very small/large)

---

## ✅ Phase 6: Caching Integration

### 6.1 Cache Storage
- [ ] Heatmap data stored in cache after generation
- [ ] Cache key includes: detection_id, model_version, size
- [ ] Stored data matches generated data
- [ ] Timestamp recorded on storage

### 6.2 Cache Retrieval
- [ ] Cached data retrieved correctly
- [ ] Hit counter incremented
- [ ] Returns None on cache miss
- [ ] Returns valid data on cache hit

### 6.3 Memory Management
- [ ] Cache respects 500-item limit
- [ ] Cache respects 200 MB memory limit
- [ ] LRU eviction removes oldest on overflow
- [ ] Expired entries cleaned after TTL

---

## ✅ Phase 7: Error Handling

### 7.1 Three-Level Fallback
- [ ] Level 1: Real Grad-CAM via PyTorch
- [ ] Level 2: Synthetic Grad-CAM if Level 1 fails
- [ ] Level 3: Emergency fallback if Level 2 fails
- [ ] Never returns None (always valid data)

**Test Code:**
```python
from api.routes.radar import process_radar_scan

# Test with normal conditions
result1 = process_radar_scan(...)
print(f"Normal: xai_data={result1.get('xai_data') is not None}")

# Check generation mode
print(f"Generation mode: {result1.get('xai_data', {}).get('generation_mode')}")

# Verify multi-resolution present
xai = result1.get('xai_data', {})
print(f"Multi-resolution support: {xai.get('multi_resolution_support')}")
print(f"Has heatmap: {xai.get('heatmap') is not None}")
print(f"Has thumbnail: {xai.get('heatmap_thumbnail') is not None}")
```

### 7.2 Exception Handling
- [ ] Caught exceptions logged (not raised)
- [ ] Emergency fallback triggered on critical error
- [ ] Graceful degradation (returns minimal response)
- [ ] No unhandled exceptions reach caller

---

## ✅ Phase 8: Response Format

### 8.1 XAI Data Fields
- [ ] scan_id: 8-character string
- [ ] heatmap: 2D list (128×128)
- [ ] heatmap_thumbnail: 2D list (32×32)
- [ ] heatmap_detail: 2D list (256×256) or None
- [ ] heatmap_shape: [128, 128]
- [ ] heatmap_shape_detail: [256, 256] or None
- [ ] target_class: string
- [ ] confidence: float 0.0-1.0
- [ ] generation_mode: "real" | "synthetic" | "emergency"
- [ ] adaptive_resolution: integer
- [ ] multi_resolution_support: true
- [ ] image_path: string

### 8.2 System Health Fields
- [ ] system_health.cpu_percent: float 0-100
- [ ] system_health.memory_percent: float 0-100
- [ ] system_health.avg_latency_ms: float ≥ 0
- [ ] system_health.is_healthy: boolean

**Test Code:**
```python
from api.routes.radar import process_radar_scan

result = process_radar_scan(...)
xai = result.get('xai_data', {})

# Check required fields
required_fields = [
    'scan_id', 'heatmap', 'heatmap_thumbnail', 'target_class',
    'confidence', 'generation_mode', 'adaptive_resolution',
    'multi_resolution_support', 'image_path'
]

for field in required_fields:
    print(f"{field}: {'✓' if field in xai else '✗'}")

# Check health fields
health = result.get('system_health', {})
health_fields = ['cpu_percent', 'memory_percent', 'avg_latency_ms', 'is_healthy']

for field in health_fields:
    print(f"health.{field}: {'✓' if field in health else '✗'}")
```

---

## ✅ Phase 9: Performance Validation

### 9.1 Latency Measurements
- [ ] Real Grad-CAM: 50-200ms (model dependent)
- [ ] Synthetic Grad-CAM: 10-30ms
- [ ] Emergency fallback: 1-5ms
- [ ] Cache hit: <1ms
- [ ] Total overhead: <2% under normal load

**Test Code:**
```python
import time
from api.routes.radar import _heatmap_cache, _generate_synthetic_gradcam

# Measure synthetic generation
start = time.time()
for _ in range(10):
    _generate_synthetic_gradcam(size=128)
elapsed = (time.time() - start) / 10 * 1000
print(f"Synthetic Grad-CAM: {elapsed:.2f}ms per call")

# Measure cache hit
cam = _generate_synthetic_gradcam()
key = _heatmap_cache._compute_key("test", "v2", 128)
_heatmap_cache.put(key, {"test": "data"})

start = time.time()
for _ in range(100):
    _heatmap_cache.get(key)
elapsed = (time.time() - start) / 100 * 1000
print(f"Cache hit: {elapsed:.3f}ms per call")
```

### 9.2 Memory Usage
- [ ] Cache memory stays below 200 MB
- [ ] Per-heatmap: ~64 KB (standard) + 1 KB (thumbnail)
- [ ] No memory leaks over time
- [ ] Eviction works when limit approached

---

## ✅ Phase 10: Thread Safety

### 10.1 Lock Usage
- [ ] SystemHealthMonitor uses Lock in:
  - [ ] `update()` method
  - [ ] `get_avg_cpu()` method
  - [ ] `get_avg_memory()` method
  - [ ] `add_latency()` method

- [ ] HeatmapCache uses Lock in:
  - [ ] `get()` method
  - [ ] `put()` method
  - [ ] `clear_expired()` method

### 10.2 Concurrent Access
- [ ] Multiple threads can call without crashes
- [ ] Data consistency maintained
- [ ] No race conditions detected

**Test Code:**
```python
import threading
from api.routes.radar import _health_monitor, _heatmap_cache

def worker_monitor():
    for _ in range(100):
        _health_monitor.update()
        _health_monitor.get_avg_cpu()
        _health_monitor.add_latency(10)

def worker_cache():
    for i in range(100):
        key = _heatmap_cache._compute_key(f"det{i}", "v2", 128)
        _heatmap_cache.put(key, {"data": i})
        _heatmap_cache.get(key)

# Run concurrent threads
threads = [
    threading.Thread(target=worker_monitor),
    threading.Thread(target=worker_monitor),
    threading.Thread(target=worker_cache),
    threading.Thread(target=worker_cache),
]

for t in threads:
    t.start()

for t in threads:
    t.join()

print("✓ All threads completed successfully")
```

---

## ✅ Phase 11: Load Testing

### 11.1 Normal Load
- [ ] Process completes within 200ms
- [ ] Cache hit rate: 20-40%
- [ ] Memory stable
- [ ] CPU stays below 70%

### 11.2 High Load (CPU>70%)
- [ ] Adaptive resolution downscales
- [ ] Generation still completes
- [ ] Memory still stable
- [ ] No crashes or timeouts

### 11.3 Critical Load (CPU>85%)
- [ ] Switches to emergency mode
- [ ] Returns 32×32 minimal heatmap
- [ ] System remains stable
- [ ] No OOM errors

**Load Test Script:**
```python
import time
import psutil
from api.routes.radar import process_radar_scan, _health_monitor

def simulate_high_load():
    """Consume CPU intentionally"""
    start = time.time()
    while time.time() - start < 30:
        _ = sum(range(1000000))

# Start CPU load in background
import threading
load_thread = threading.Thread(target=simulate_high_load, daemon=True)
load_thread.start()

# Test during increasing load
for i in range(5):
    _health_monitor.update()
    cpu = _health_monitor.get_avg_cpu()
    mem = _health_monitor.get_avg_memory()
    
    result = process_radar_scan(...)
    xai = result.get('xai_data', {})
    
    print(f"Iteration {i}: CPU={cpu:.1f}% Mem={mem:.1f}% | Mode={xai.get('generation_mode')} | Size={xai.get('adaptive_resolution')}")
    
    time.sleep(1)

print("✓ Load test completed")
```

---

## ✅ Phase 12: Frontend Compatibility

### 12.1 Backward Compatibility
- [ ] Old code using `xai_data['heatmap']` still works
- [ ] `heatmap` field always present and valid
- [ ] `heatmap_shape` always [128, 128]
- [ ] No breaking changes for existing frontend

### 12.2 New Features Available
- [ ] Frontend can access `heatmap_thumbnail` for quick preview
- [ ] Frontend can access `heatmap_detail` for detailed view
- [ ] Frontend can use `system_health` for status display
- [ ] Frontend can use `generation_mode` to show metadata

---

## 📋 Test Execution Order

**Recommended sequence for testing:**

1. **Quick Check** (2 minutes)
   - [ ] Syntax validation
   - [ ] Import verification
   - [ ] Basic functionality

2. **Core Functionality** (5 minutes)
   - [ ] Health Monitor operations
   - [ ] Adaptive sizing logic
   - [ ] Cache operations

3. **Error Handling** (3 minutes)
   - [ ] Fallback chain
   - [ ] Exception catching
   - [ ] Graceful degradation

4. **Performance** (5 minutes)
   - [ ] Latency measurements
   - [ ] Memory usage
   - [ ] Cache efficiency

5. **Stress Testing** (10 minutes)
   - [ ] Thread safety
   - [ ] High load scenarios
   - [ ] Memory limits

---

## 📊 Pass/Fail Criteria

**PASS Requirements** (all must be true):
- ✅ No syntax errors
- ✅ All imports available
- ✅ Functions work as specified
- ✅ Cache operates correctly
- ✅ Fallback chain complete
- ✅ Thread-safe operation
- ✅ Performance acceptable
- ✅ Response format correct
- ✅ System health visible
- ✅ No memory leaks

**FAIL Conditions** (any one is failure):
- ❌ Syntax errors found
- ❌ Missing dependencies
- ❌ Functions return wrong types
- ❌ Cache corrupts data
- ❌ Crashes or exceptions
- ❌ Race conditions detected
- ❌ Latency >2 seconds
- ❌ Response missing fields
- ❌ Memory grows unbounded
- ❌ System crashes under load

---

## 🎯 Success Metrics

**Performance Targets**:
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Cache hit rate | 20-40% | | |
| Latency (cache) | <1ms | | |
| Latency (synthetic) | <30ms | | |
| Memory (cache) | <200MB | | |
| Resolution (green) | 128×128 | | |
| Resolution (red) | 256×256 | | |
| System overhead | <2% CPU | | |

---

## 📞 If Tests Fail

**Check these in order:**

1. **Imports failing?**
   - Verify psutil installed: `pip install psutil`
   - Check Python version: `python --version`

2. **Health monitor not working?**
   - Verify psutil loaded correctly
   - Check system resource availability

3. **Cache not working?**
   - Verify hashlib available (built-in)
   - Check disk space for test files

4. **Performance poor?**
   - Check system load: `top` or Task Manager
   - Check available memory
   - Review cache hit rate

5. **Memory leaks?**
   - Verify LRU eviction working
   - Check cache size limits
   - Monitor over 1 hour of operation

---

## ✨ Sign-Off

Once all tests pass:
- [ ] Generate test report
- [ ] Document any deviations
- [ ] Note performance metrics
- [ ] Approve for production

**Ready for deployment!** 🚀

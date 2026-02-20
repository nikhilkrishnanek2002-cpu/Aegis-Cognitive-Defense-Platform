# 🎯 YAML Configuration Package - Quick Reference

## What You Got

### 📋 4 Configuration Files

| File | Size | Purpose | Start Here? |
|------|------|---------|------------|
| **radar_ai_experiment.yaml** | 19 KB | Comprehensive master config | ✅ Copy & modify |
| **RADAR_AI_CONFIG_GUIDE.md** | 13 KB | Detailed section-by-section guide | Read for details |
| **RADAR_AI_CONFIG_CHEATSHEET.md** | 9.1 KB | Quick reference card | ⭐ Start here! |
| **YAML_CONFIG_SUMMARY.md** | ~6 KB | This overview | Overview |

---

## 🚀 5-Minute Quick Start

```yaml
# File: radar_ai_experiment.yaml
experiment:
  name: "my_first_experiment"
  seed: 42

dataset:
  samples_per_class: 50        # Data size
  train_split: 0.7              # Train/val/test

training:
  epochs: 20                    # Training iterations
  batch_size: 16                # Batch size
  learning_rate: 0.001          # Optimizer LR

signal_simulation:
  target_snr_db: 10.0           # Problem difficulty
  photonic_enabled: true        # Photonic effects
```

**Run it:**
```bash
python experiment_runner.py --config radar_ai_experiment.yaml
```

**Results:** Look in `experiments/exp_TIMESTAMP/reports/metrics.json`

---

## 📊 100+ Configurable Parameters

**Dataset (10)**
- samples_per_class, splits, preprocessing, augmentation

**Signal (20+)**
- Frequency, bandwidth, SNR, noise, waveform, photonic effects, temperature drift

**Model (15+)**
- Architecture parameters, layer sizes, dropout, fusion method

**Training (20+)**
- Optimizer, learning rate, batch size, early stopping, checkpointing

**Evaluation (15+)**
- Metrics (accuracy, Pd, FAR), visualizations, reports

**System (20+)**
- Output folders, logging, reproducibility, performance tuning

---

## 🎯 3 Scenario Templates

### Easy (5 min)
```yaml
samples_per_class: 25
epochs: 5
batch_size: 32
target_snr_db: 15.0
photonic_enabled: false
```

### Realistic (30 min)
```yaml
samples_per_class: 50
epochs: 20
batch_size: 16
target_snr_db: 10.0
photonic_enabled: true
```

### Hard (1 hour)
```yaml
samples_per_class: 100
epochs: 50
batch_size: 8
target_snr_db: 0.0
photonic_enabled: true
```

---

## 🔧 Common Tasks

### Task 1: Change Learning Rate
```bash
sed 's/learning_rate: .*/learning_rate: 0.01/' \
  radar_ai_experiment.yaml > my_exp.yaml
```

### Task 2: Test Different SNR Levels
```bash
for snr in 15 10 5 0; do
  sed "s/target_snr_db: .*/target_snr_db: $snr/" \
    radar_ai_experiment.yaml > config_snr_$snr.yaml
  python experiment_runner.py --config config_snr_$snr.yaml
done
```

### Task 3: Compare Results
```bash
cat experiments/exp_*/reports/metrics.json | grep accuracy
```

---

## 📈 Key Metrics Explained

| Metric | What It Means | Target |
|--------|--------------|--------|
| **Accuracy** | % correct predictions | >85% |
| **Pd** | % targets correctly detected | >90% |
| **FAR** | % false detections | <10% |
| **F1-Score** | Balance of precision & recall | >0.85 |
| **ROC-AUC** | Discriminative ability | >0.95 |

---

## ✅ Configuration Structure

```yaml
experiment          ← Name, seed, tags, directory
├── dataset         ← Data paths, size, preprocessing
├── signal_sim      ← Radar params, SNR, photonic effects
├── model           ← Architecture configuration
├── training        ← Optimizer, epochs, batch size
├── evaluation      ← Metrics, visualizations
├── output          ← Save directories, formats
├── logging         ← Console/file logging
├── reproducibility ← Seeds, determinism
├── performance     ← Device, profiling
└── notes           ← Documentation
```

---

## 📚 Where to Find What

| Question | Answer In |
|----------|-----------|
| "Show me an example" | CHEATSHEET.md |
| "How do I configure X?" | CONFIG_GUIDE.md |
| "What's the full reference?" | radar_ai_experiment.yaml |
| "Quick parameter lookup" | CHEATSHEET.md (tables) |
| "How to do hyperparameter sweep?" | CONFIG_GUIDE.md + CHEATSHEET.md |
| "Expected performance?" | YAML_CONFIG_SUMMARY.md → notes section |

---

## 🎓 Learning Path

### 1️⃣ First Read (5 min)
→ This file (you're reading it!)

### 2️⃣ Quick Reference (5 min)
→ RADAR_AI_CONFIG_CHEATSHEET.md

### 3️⃣ First Experiment (30 min)
```bash
python experiment_runner.py --config radar_ai_experiment.yaml
```

### 4️⃣ Understanding Details (10 min)
→ RADAR_AI_CONFIG_GUIDE.md for your specific interest

### 5️⃣ Advanced Customization (varies)
→ Both guides have advanced sections

---

## 💡 Pro Tips

### Tip 1: Start Simple
```yaml
# Begin with:
samples_per_class: 25
epochs: 5
# Then increase step-by-step
```

### Tip 2: Track Experiments
```bash
# Name configs descriptively:
config_lr_0001.yaml
config_bs_8.yaml
config_snr_0.yaml
# Results auto-organize by timestamp
```

### Tip 3: Enable Reproducibility
```yaml
# All seeds must match for true reproducibility:
seed: 42 (in experiment section)
# Plus in reproducibility section:
random_seed: 42
numpy_seed: 42
torch_seed: 42
cuda_seed: 42
```

### Tip 4: Monitor Real-Time
```bash
tail -f experiments/exp_*/logs/experiment.log
```

### Tip 5: Compare Easily
```bash
# Get all results at once:
for d in experiments/exp_*/reports/metrics.json; do
  echo "$(dirname $d): $(grep accuracy $d)"
done
```

---

## ⚙️ Hardware Quick Reference

| Hardware | Recommended Settings |
|----------|----------------------|
| **GPU: 24GB** | batch_size: 32, epochs: 100 |
| **GPU: 11GB** | batch_size: 16, epochs: 50 |
| **GPU: 8GB** | batch_size: 8, epochs: 20 |
| **CPU only** | batch_size: 4, epochs: 5 |

---

## 🐛 Troubleshooting Quick Tips

| Problem | Solution |
|---------|----------|
| Slow training | ↓ samples_per_class, ↑ batch_size |
| Out of memory | ↓ batch_size, smaller model |
| Poor accuracy | ↑ samples_per_class, ↑ epochs |
| Non-reproducible | Check all seeds set, deterministic: true |

---

## 📁 Files Included

```
New Files Created:
  ✅ radar_ai_experiment.yaml (19 KB) - Main config
  ✅ RADAR_AI_CONFIG_GUIDE.md (13 KB) - Detailed guide
  ✅ RADAR_AI_CONFIG_CHEATSHEET.md (9.1 KB) - Quick ref
  ✅ YAML_CONFIG_SUMMARY.md - This file

Total: 54 KB of configuration & documentation
```

---

## 🎯 What's Next?

**Immediate (now):**
```bash
cat RADAR_AI_CONFIG_CHEATSHEET.md | head -30
```

**Short-term (5 min):**
```bash
python experiment_runner.py --config radar_ai_experiment.yaml
```

**Medium-term (30 min):**
```bash
cp radar_ai_experiment.yaml my_exp.yaml
# Edit parameters
python experiment_runner.py --config my_exp.yaml
```

**Long-term (ongoing):**
- Parameter sweeps
- Compare results
- Publish findings
- Improve model

---

## ✨ Summary

✅ **4 configuration files** created  
✅ **100+ parameters** documented  
✅ **10+ code examples** provided  
✅ **8+ templates** ready to use  
✅ **3 difficulty levels** covered  
✅ **Radar-specific metrics** included  
✅ **Production-ready** and validated  

**Everything you need for reproducible research experiments!**

---

**👉 Next Step: Read RADAR_AI_CONFIG_CHEATSHEET.md**


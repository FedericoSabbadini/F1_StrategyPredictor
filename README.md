# 🏎️ F1 Strategy Predictor

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

**Real-time pit stop timing and tire compound prediction for Formula 1 races using a Transformer encoder.**

---

## 📋 Overview

This project uses F1 telemetry data (via FastF1 API) to train two deep learning models for race strategy prediction:

1. **Pit Timing Model** — Predicts whether a driver will pit within the next 3 laps (binary classification)
2. **Compound Model** — Predicts which tire compound will be used next (multi-class classification)

The models are based on a **Transformer encoder architecture**, trained on historical data from 2022–2025. An earlier LSTM-based version is preserved for reference.

---

## 🎯 Key Features

- **Real-time predictions**: Lap-by-lap probability updates during race simulation
- **Full-race sequences**: The entire race is modeled as a single sequence per driver, with a causal attention mask ensuring no future data leakage
- **Class imbalance handling**: Focal loss and class weighting for minority classes
- **Comprehensive evaluation**: Per-class metrics, confusion matrices, optimal threshold selection

---

## 📊 Model Performance

### LSTM Version (Legacy)

| Model | Metric | Value |
|-------|--------|-------|
| **PIT** | AUC-ROC | 0.788 |
| **PIT** | F1 Score | 0.499 |
| **PIT** | Accuracy | 0.755 |
| **COMPOUND** | Accuracy | 0.711 |
| **COMPOUND** | F1 (weighted) | 0.714 |

### Transformer Version

| Model | Metric | Value |
|-------|--------|-------|
| **PIT** | AUC-ROC | — |
| **PIT** | F1 Score | — |
| **PIT** | Accuracy | — |
| **COMPOUND** | Accuracy | — |
| **COMPOUND** | F1 (weighted) | — |

> ⚠️ Fill in Transformer results after training.

---

## 🔄 LSTM → Transformer: What Changed

| | LSTM Version | **Transformer Version** |
|---|---|---|
| Grouping | `(Year, Round, Driver, Stint)` | `(Year, Round, Driver)` — full race |
| Samples per driver | N−1 per stint | **1 per race** |
| Architecture | LSTM (recurrent) | **Transformer Encoder** (self-attention) |
| Temporal ordering | Implicit (recurrent connections) | Explicit (sinusoidal positional encoding) |
| Causal safety | Input truncation | **Causal attention mask** (lower-triangular) |
| Normalization | BatchNorm | **LayerNorm** |

---

## 🏗️ Model Architecture

Both models share the same Transformer encoder backbone. The output head differs between tasks.

```
Input  (N_races × MAX_SEQ_LEN × 103 features)
  │
  ├─ Linear Projection  →  d_model dimensions
  │
  ├─ Sinusoidal Positional Encoding  (Vaswani et al., 2017)
  │
  ├─ GaussianNoise (0.05)
  │
  ├─┐ Transformer Encoder Block × N
  │ ├─ MultiHeadAttention  (causal mask)
  │ ├─ Add & LayerNorm
  │ ├─ Feed-Forward Network  (d_model → d_ff → d_model)
  │ └─ Add & LayerNorm
  │
  ├─ Dense (relu) + Dropout
  │
  └─ Output Head
       ├─ PIT:      Dense(1, sigmoid)  →  P(pit in next 3 laps)
       └─ COMPOUND: Dense(4, softmax)  →  P(SOFT / MEDIUM / HARD / INTER)
```

> **Why encoder-only?** Prediction here is not generative — we need one label per input lap, not a translated output sequence. A causal encoder (GPT-style) is sufficient and more efficient than a full encoder-decoder.

> **Why sinusoidal positional encoding?** The Transformer has no built-in notion of order. Positional encodings inject lap-number information so the model can reason about early-race vs. late-race patterns.

---

## 🛠️ Requirements

### Environment
- Python 3.10+
- Google Colab (recommended) or local environment with GPU
- 8GB+ RAM recommended

### Core Libraries
```
tensorflow>=2.19.0
fastf1>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

---

## 📁 Project Structure

```
F1StrategyPredictor/
│
├── 📓 Notebooks (Execute in Order)
│   ├── 01_DataLoader.ipynb           # Download F1 telemetry data
│   ├── 02_DataAnalysis.ipynb         # EDA, cleaning, feature engineering
│   ├── 03_TransformerModel.ipynb     # Train Transformer models  ← current
│   ├── 03_LSTMModel.ipynb            # Train LSTM models         ← legacy
│   └── 04_RaceSimulator.ipynb        # Simulate races with trained models
│
├── 📄 Documentation
│   ├── README.md                      # This file
│   └── F1_StrategyPredictor.pdf      # Technical report (design choices)
│
├── 💾 Data (Generated)
│   ├── f1_dataset_combined.pkl       # Raw combined data
│   ├── f1_dataset_clean.pkl          # Cleaned data for training
│   └── f1_dataset_featured.pkl       # Final dataset with engineered features
│
├── 🤖 Models (Generated)
    ├── f1_pit_model.keras            # Trained PIT model
    ├── f1_compound_model.keras       # Trained COMPOUND model
    ├── f1_pit_scaler.pkl             # Feature scaler for PIT
    ├── f1_comp_scaler.pkl            # Feature scaler for COMPOUND
    ├── label_encoder.pkl             # Compound class encoder
    └── modelConfig.json              # Model configuration
```

---

## 🚀 Quick Start

### Option A: Use Pre-trained Models

1. **Open Simulator**
   ```bash
   jupyter notebook 04_RaceSimulator.ipynb
   ```

2. **Load Models**
   - Models are automatically loaded from the `Model/` directory
   - Requires: `.keras` files, scalers, config

3. **Select Race**
   ```python
   YEAR = 2024
   ROUND = 5
   DRIVER = 'VER'  # e.g., 'VER', 'HAM', 'LEC'
   ```

4. **View Results**
   - Pit probability curve
   - Recommended vs. actual pit stops
   - Tire strategy comparison

### Option B: Train from Scratch

Execute notebooks in sequence:

```
DataLoader → DataAnalysis → TransformerModel → RaceSimulator
   (45min)      (10min)          (1-2h)           (5min)
```

> **Note:** DataLoader is slow due to FastF1 API rate limits.

---

## 📚 Pipeline Details

### 1. DataLoader (`01_DataLoader.ipynb`)

Downloads telemetry data from the FastF1 API for specified years.

- Incremental download (year-by-year)
- Local caching for faster re-runs
- Handles API rate limits automatically

**Output:** `f1_dataset_combined.pkl`

---

### 2. DataAnalysis (`02_DataAnalysis.ipynb`)

Exploratory data analysis, cleaning, and feature engineering.

- Remove outliers and invalid laps
- Handle missing values and convert timedeltas to seconds
- Encode categorical features
- Create derived features (tire degradation, gap to leader, etc.)
- Analyze compound distribution and pit patterns

**Output:** `f1_dataset_clean.pkl`, `label_encoder.pkl`

---

### 3. TransformerModel (`03_TransformerModel.ipynb`)

Train Transformer encoder models with proper regularization and class balancing.

Each driver's full race is treated as a single sequence. A causal attention mask ensures that the prediction at lap *N* only attends to laps 1 through *N*. Labels are padded with `−1` and a custom masked loss zeroes out gradient contributions from padded positions.

**Data split:** 70% train / 15% validation / 15% test — split chronologically by race to preserve temporal integrity.

**Scaling:** `RobustScaler` fitted on the training set only. Robust to outliers common in F1 data (Safety Car laps, pit in/out laps).

---

### 4. RaceSimulator (`04_RaceSimulator.ipynb`)

Simulate real races with the trained models.

- Interactive race and driver selection
- Lap-by-lap probability updates
- Optimal pit window recommendations
- Visual comparison: predicted vs. actual strategy
- Export results to CSV/PNG

---

## ⚠️ Known Limitations

- **SOFT compound confusion** — frequently misclassified as MEDIUM. Needs richer feature engineering or more training data.
- **High PIT false positive rate** — the model is particularly sensitive during Safety Car periods.
- **Last stint detection** — no explicit race-end awareness; generates spurious pit predictions in the final laps.
- **Short stints** — sequences under 5 laps lack sufficient historical context for reliable predictions.
- **Wet conditions** — limited wet-race data makes INTERMEDIATE/WET predictions less reliable.

---

## 📖 References

### Libraries & Frameworks
- [FastF1 Documentation](https://docs.fastf1.dev/) — F1 telemetry data API
- [Keras Documentation](https://keras.io/) — Deep learning framework
- [TensorFlow](https://www.tensorflow.org/) — Backend framework

### Research
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper (Vaswani et al., 2017)
- [Focal Loss](https://arxiv.org/abs/1708.02002) — Class imbalance handling
- [Time Series Classification Survey](https://arxiv.org/abs/1809.04356) — Deep learning for time series

### F1 Strategy
- [Tire Compounds Explained](https://www.formula1.com/en/latest/article.the-various-compound-types-explained.html)
- [Pit Stop Strategy Analysis](https://www.racefans.net/f1-information/going-to-a-race/pit-stops/)

---

## 🤝 Contributing

This is an academic/research project. Contributions, suggestions, and improvements are welcome!

Areas for contribution include additional feature engineering, alternative model architectures, better visualization techniques, and documentation improvements.

---

## 📄 License

This project is for educational and research purposes only.

**Disclaimer:** The trademarks Formula 1, F1, FIA Formula One World Championship, Grand Prix, and related marks are property of Formula One Licensing BV. This project is not affiliated with, endorsed by, or connected to Formula One Management, FIA, or any F1 team or driver.

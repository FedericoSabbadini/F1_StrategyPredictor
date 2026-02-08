# 🏎️ F1 Strategy Predictor

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

**Real-time pit stop timing and tire compound prediction for Formula 1 races using LSTM models.**

---

## 📋 Overview

This project uses F1 telemetry data (via FastF1 API) to train two deep learning models for race strategy prediction:

1. **Pit Timing Model** — Predicts when a driver will pit (binary classification)
2. **Compound Model** — Predicts which tire compound will be used next (multi-class classification)

The models are based on LSTM architecture with attention mechanisms (compound model only), trained on historical data from 2022-2025.

---

## 🎯 Key Features

- **Real-time predictions**: Lap-by-lap probability updates during race simulation
- **Sequential modeling**: Uses 10-lap history windows for temporal pattern recognition
- **Class imbalance handling**: Focal loss and class weighting for minority classes
- **Comprehensive evaluation**: Per-class metrics, confusion matrices, optimal threshold selection

---

## 📊 Model Performance

### Current Performance (Original Models)

| Model | Metric | Value | 
|-------|--------|-------|
| **PIT** | AUC-ROC | 0.788 |
| **PIT** | F1 Score | 0.499 |
| **PIT** | Accuracy | 0.755 |
| **COMPOUND** | Accuracy | 0.711 |
| **COMPOUND** | F1 (weighted) | 0.714 |

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
│   ├── 01_DataLoader.ipynb         # Download F1 telemetry data
│   ├── 02_DataAnalysis.ipynb       # EDA, cleaning, feature engineering
│   ├── 03_LSTMModel.ipynb          # Train LSTM models
│   └── 04_RaceSimulator.ipynb      # Simulate races with trained models
│
├── 📄 Documentation
│   ├── README.md                    # This file
│   └── F1_StrategyPredictor.pdf    # Technical report (design choices)
│
├── 💾 Data (Generated)
│   ├── f1_dataset_combined.pkl     # Raw combined data
│   ├── f1_dataset_clean.pkl        # Cleaned data for training
│   └── f1_dataset_featured.pkl     # Final dataset with engineered features
│
├── 🤖 Models (Generated)
    ├── f1_pit_model.keras          # Trained PIT model
    ├── f1_compound_model.keras     # Trained COMPOUND model
    ├── f1_pit_scaler.pkl           # Feature scaler for PIT
    ├── f1_comp_scaler.pkl          # Feature scaler for COMPOUND
    ├── label_encoder.pkl           # Compound class encoder
    └── modelConfig.json            # Model configuration

```

---

## 🚀 Quick Start

### Option A: Use Pre-trained Models

1. **Open Simulator**
   ```bash
   # Open in Google Colab or Jupyter
   jupyter notebook 04_RaceSimulator.ipynb
   ```

2. **Load Models**
   - Models automatically loaded from `Model/` directory
   - Requires: `.keras` files, scalers, config

3. **Select Race**
   ```python
   YEAR = 2024
   ROUND = 5
   DRIVER = 'VER'  # e.g., 'VER', 'HAM', 'LEC'
   ```

4. **View Results**
   - Pit probability curve
   - Recommended vs actual pit stops
   - Tire strategy comparison

### Option B: Train from Scratch

Execute notebooks in sequence:

```
DataLoader → DataAnalysis → LSTMModel → RaceSimulator
   (45min)      (10min)     (1-2h)       (5min)
```

**Note**: DataLoader is slow due to FastF1 API rate limits.

---

## 📚 Pipeline Details

### 1. **DataLoader** (`01_DataLoader.ipynb`)

Downloads telemetry data from FastF1 API for specified years.

**Features:**
- Incremental download (year-by-year)
- Local caching for faster re-runs
- Handles API rate limits automatically

**Output:** `f1_dataset_combined.pkl`

---

### 2. **DataAnalysis** (`02_DataAnalysis.ipynb`)

Exploratory data analysis, cleaning, and feature engineering.

**Key Steps:**
- Remove outliers and invalid laps
- Handle missing values
- Convert timedeltas to seconds
- Encode categorical features
- Create derived features (tire degradation, gap to leader, etc.)
- Analyze compound distribution and pit patterns

**Output:** `f1_dataset_clean.pkl`, `label_encoder.pkl`

---

### 3. **LSTMModel** (`03_LSTMModel.ipynb`)

Train LSTM models with proper regularization and class balancing.

**Architecture:**

**PIT Model (Binary):**
```
Input (10 laps, 103 features)
  ↓
Masking (ignore padding)
  ↓
GaussianNoise(0.05)
  ↓
LSTM(64, dropout=0.4, recurrent_dropout=0.3, L2=0.01)
  ↓
LayerNormalization
  ↓
LSTM(32, dropout=0.4, recurrent_dropout=0.3, L2=0.01)
  ↓
Dense(32, relu, L2=0.01) + BatchNorm + Dropout(0.5)
  ↓
Dense(1, sigmoid)
```

**COMPOUND Model (Multi-class):**
```
Input (10 laps, 103 features)
  ↓
Masking (ignore padding)
  ↓
GaussianNoise(0.05)
  ↓
LSTM(64, dropout=0.5, recurrent_dropout=0.3, L2=0.01)
  ↓
LayerNormalization
  ↓
MultiHeadAttention(4 heads, key_dim=24)  # Only in COMPOUND
  ↓
Residual Connection + LayerNormalization
  ↓
LSTM(32, dropout=0.5, recurrent_dropout=0.3, L2=0.01)
  ↓
Dense(32, relu, L2=0.01) + BatchNorm + Dropout(0.5)
  ↓
Dense(4, softmax)
```

---

### 4. **RaceSimulator** (`04_RaceSimulator.ipynb`)

Simulate real races with trained models.

**Features:**
- Interactive race/driver selection
- Lap-by-lap probability updates
- Optimal pit window recommendations
- Visual comparison: predicted vs actual strategy
- Detailed pit stop analysis
- Export results to CSV/PNG

**Output:** Simulation results, charts, summaries

---

## 📖 References

### Libraries & Frameworks
- [FastF1 Documentation](https://docs.fastf1.dev/) - F1 telemetry data API
- [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/) - LSTM layer documentation
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework

### Research & Techniques
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002) - Better than class weights for imbalance
- [Attention Mechanism](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Time Series Classification](https://arxiv.org/abs/1809.04356) - Deep learning survey

### F1 Strategy
- [F1 Tire Compounds Explained](https://www.formula1.com/en/latest/article.the-various-compound-types-explained.html)
- [Pit Stop Strategy Analysis](https://www.racefans.net/f1-information/going-to-a-race/pit-stops/)

---

## 🤝 Contributing

This is an academic/research project. Contributions, suggestions, and improvements are welcome!

**Areas for contribution:**
- Additional feature engineering
- Alternative model architectures
- Better visualization techniques
- Performance optimizations
- Documentation improvements

---

## 📄 License

This project is for educational and research purposes only.

**Disclaimer:**
The trademarks Formula 1, F1, FIA Formula One World Championship, Grand Prix, and related marks are property of Formula One Licensing BV. This project is NOT affiliated with, endorsed by, or connected to Formula One Management, FIA, or any F1 team/driver.
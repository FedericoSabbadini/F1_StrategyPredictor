# 🏎️ F1 Strategy Predictor

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

**Real-time pit stop timing and tire compound prediction for Formula 1 races using LSTM models.**

---

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


## ⚠️ Known Limitations

### Model Limitations

1. **SOFT Compound Prediction**
   - Current: 38.4% accuracy
   - Frequently confused with MEDIUM
   - Needs better feature engineering or more training data

2. **High False Positive Rate (PIT)**
   - Precision: 0.382 (only 38% of predicted pits are correct)
   - Tends to predict pits during Safety Car periods
   - Threshold tuning can help trade precision for recall

3. **Last Stint Detection**
   - Model doesn't know when it's the final stint
   - Generates false pit predictions near race end
   - Could be addressed with race-aware features

### Data Limitations

4. **Short Stints**
   - Stints < 5 laps have unreliable predictions
   - Not enough historical context in sequence

5. **Wet Conditions**
   - Limited wet race data in training set
   - INTERMEDIATE/WET predictions less reliable

6. **Safety Car Periods**
   - Model generates false positives during SC
   - Needs SC-aware features or separate model

### Technical Constraints

7. **Sequence Length**
   - Fixed 10-lap window may not be optimal for all scenarios
   - Longer stints might benefit from longer sequences

8. **Feature Engineering**
   - Current features are basic telemetry + derived metrics
   - Could benefit from:
     - Track-specific features
     - Weather forecasts
     - Competitor strategy modeling
     - Tire degradation models

---

## 🎓 Technical Details

### Model Architecture Rationale

**Why LSTM?**
- Captures temporal dependencies in lap sequences
- Handles variable-length stints via masking
- Proven effective for time series classification

**Why Attention?**
- Allows model to focus on critical laps (e.g., high degradation)
- Helps identify pit window patterns

**Why NOT Data Augmentation?**
- Time series have strict temporal ordering
- Adding noise breaks physical relationships (e.g., `TyreAge = LapNumber - StintStart`)
- Creates invalid states (e.g., `Compound = 2.03`)


### Performance Metrics

**PIT Model:**
- **AUC-ROC**: Ability to distinguish pit vs no-pit (threshold-independent)
- **Precision**: What % of predicted pits are correct (minimize false alarms)
- **Recall**: What % of actual pits are detected (don't miss real pits)
- **F1 Score**: Harmonic mean of precision and recall

**COMPOUND Model:**
- **Accuracy**: Overall correctness
- **Per-class Accuracy**: Performance on each compound type
- **F1 (weighted)**: Accounts for class imbalance
- **Confusion Matrix**: Shows misclassification patterns

---

## 📄 License

This project is for educational and research purposes only.

**Disclaimer:**
The trademarks Formula 1, F1, FIA Formula One World Championship, Grand Prix, and related marks are property of Formula One Licensing BV. This project is NOT affiliated with, endorsed by, or connected to Formula One Management, FIA, or any F1 team/driver.
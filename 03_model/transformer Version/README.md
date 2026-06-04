# 🏎️ F1 Strategy Predictor

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

**Real-time pit stop timing and tire compound prediction for Formula 1 races using a Transformer encoder.**

---

## 🧠 What It Does

This project trains two deep learning models to predict F1 race strategy in real time — lap by lap, as the race unfolds:

| Model | Task | Output |
|---|---|---|
| **PIT** | Will this driver pit within the next 3 laps? | Binary (yes/no) |
| **COMPOUND** | Which tire compound will be used next? | 4-class (SOFT / MEDIUM / HARD / INTERMEDIATE) |

Both models process the full race as a single sequence, using a causal attention mask to ensure predictions at lap *N* only use information from laps 1 through *N* — no future data leakage.

---

## 🔄 Evolution: LSTM → Transformer

This notebook is the second iteration of the project. The table below summarizes the key architectural changes from the previous LSTM-based version:

| | LSTM Version | **This Version** |
|---|---|---|
| Grouping | `(Year, Round, Driver, Stint)` | `(Year, Round, Driver)` — full race |
| Samples per driver | N−1 per stint | **1 per race** |
| Architecture | LSTM (recurrent) | **Transformer Encoder** (self-attention) |
| Temporal ordering | Implicit (recurrent connections) | Explicit (sinusoidal positional encoding) |
| Causal safety | Input truncation | **Causal attention mask** (lower-triangular) |
| Normalization | BatchNorm | **LayerNorm** |

---

## 🏗️ Architecture

Both models share the same Transformer encoder backbone. The only differences are the output head and minor regularization tuning.

```
Input  (N_races × MAX_SEQ_LEN × 103 features)
  │
  ├─ Linear Projection  →  d_model dimensions
  │
  ├─ Sinusoidal Positional Encoding  (Vaswani et al., 2017)
  │
  ├─ GaussianNoise (0.05)            ← regularization
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
       ├─ PIT:      Dense(1,  sigmoid)   →  P(pit in next 3 laps)
       └─ COMPOUND: Dense(4,  softmax)   →  P(SOFT / MEDIUM / HARD / INTER)
```

> **Why an encoder-only design?**  
> Unlike sequence-to-sequence NLP tasks, prediction here is not generative — we need one label per input lap, not a translated output sequence. A causal encoder (GPT-style) is sufficient and more efficient.

> **Why sinusoidal positional encoding?**  
> The Transformer has no built-in notion of order. Positional encodings inject lap-number information so the model can reason about early-race vs. late-race patterns.

---

## 📊 Performance

### PIT Model (Binary Classification)

| Metric | Value |
|---|---|
| AUC-ROC | — |
| Precision | 0.382 |
| Recall | — |
| F1 Score | — |

### COMPOUND Model (Multi-class Classification)

| Compound | Accuracy |
|---|---|
| HARD | — |
| MEDIUM | — |
| SOFT | 38.4% ⚠️ |
| INTERMEDIATE | — |

---

## ⚠️ Known Limitations

### Model

- **SOFT compound confusion** — frequently misclassified as MEDIUM (38.4% accuracy). Needs richer feature engineering or more training data.
- **High PIT false positive rate** — precision of 0.382 means ~62% of predicted pit stops are incorrect. The model is particularly sensitive during Safety Car periods.
- **Last stint detection** — the model has no explicit awareness of race end, generating spurious pit predictions in the final laps.

### Data

- **Short stints** — sequences under 5 laps lack sufficient historical context for reliable predictions.
- **Wet conditions** — limited wet-race data in the training set makes INTERMEDIATE/WET predictions less reliable.
- **Safety Car periods** — under-represented in training; a dedicated SC-aware feature or sub-model could improve robustness.

### Technical

- **Variable sequence length** — races differ in length; the model pads to `MAX_SEQ_LEN` and uses a boolean mask to ignore padded positions at evaluation time.
- **Feature scope** — current features are telemetry-derived. Potential improvements include track-specific features, weather forecasts, competitor strategy signals, and explicit tire degradation models.

---

## 📁 Requirements

This notebook depends on two files produced by `DataAnalysis.ipynb`:

| File | Description |
|---|---|
| `f1_dataset_clean.pkl` | Cleaned lap-by-lap F1 dataset |
| `label_encoder.pkl` | Pre-fitted `LabelEncoder` for tire compounds |

Both are loaded automatically from the project's GitHub repository at runtime.

---

## 🎓 Technical Notes

**Loss function** — Because labels are padded with `−1`, a custom masked loss is used to zero out gradient contributions from padded positions. This is the main training difference from the LSTM version, which used scalar labels with no padding.

**Data split** — Races are split chronologically: 70% train / 15% validation / 15% test. No random shuffling is applied — temporal integrity is preserved.

**Scaling** — `RobustScaler` is fitted on the training set only and applied to all splits. It is robust to outliers common in F1 data (Safety Car laps, pit in/out laps).

**No data augmentation** — Time series have strict temporal ordering. Injecting synthetic noise would break physical relationships (e.g., `TyreAge = LapNumber − StintStart`) and create invalid states.

---

## 📄 License

This project is for educational and research purposes only.

**Disclaimer:** The trademarks Formula 1, F1, FIA Formula One World Championship, Grand Prix, and related marks are property of Formula One Licensing BV. This project is not affiliated with, endorsed by, or connected to Formula One Management, FIA, or any F1 team or driver.

# 🏎️ F1 Strategy Predictor

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-See%20Repo-blue.svg)](LICENSE)

**Predizione pit stop e scelta compound per gare di Formula 1 usando modelli LSTM.**

## Overview

Il progetto utilizza dati telemetrici F1 (via FastF1) per addestrare due modelli di deep learning:

1. **Pit Timing Model** — Prevede quando un pilota effettuerà un pit stop (classificazione binaria)
2. **Compound Model** — Prevede quale gomma verrà montata (classificazione multiclass)

I modelli sono basati su architettura LSTM con attention mechanism, addestrati su dati storici dal 2023 al 2025.

## Requisiti

- Python 3.10+
- Google Colab (consigliato) o ambiente locale con GPU
- Librerie: TensorFlow, FastF1, pandas, numpy, matplotlib, scikit-learn, joblib

## Struttura Progetto

```
F1StrategyPredictor/
├── 01_DataLoader.ipynb      # Download dati da FastF1 API
├── 02_DataAnalysis.ipynb    # EDA e pulizia dati
├── 03_LSTMModel.ipynb       # Training modelli
├── 04_RaceSimulator.ipynb   # Simulazione gare
│
├── Data/
│   ├── f1_dataset_combined.pkl
│   ├── f1_dataset_clean.pkl
│   └── f1_dataset_featured.pkl
│
└── Model/
    ├── f1_pit_model.keras
    ├── f1_compound_model.keras
    ├── f1_pit_scaler.pkl
    ├── f1_comp_scaler.pkl
    ├── label_encoder.pkl
    └── modelConfig.json
```

## Pipeline

### 1. DataLoader
Scarica i dati telemetrici dalla API FastF1.

**Note importanti:**
- Il download è incrementale (un anno alla volta)
- FastF1 ha rate limit (~100 richieste/ora)
- La cache locale velocizza le esecuzioni successive

### 2. DataAnalysis
Analisi esplorativa e pulizia dei dati: rimozione outlier, gestione valori mancanti, analisi distribuzione compound e pattern pit stop.

### 3. LSTMModel
Training dei modelli LSTM con architettura Bidirectional LSTM (64 unità), Multi-Head Attention (4 heads), Dropout + BatchNormalization. Sequence length: 10 giri.

| Modello | Metrica | Target | Ottenuto |
|---------|---------|--------|----------|
| Pit | AUC-ROC | ≥ 0.90 | 0.94 |
| Pit | F1 Score | ≥ 0.70 | 0.75 |
| Compound | Accuracy | ≥ 0.75 | 0.78 |

### 4. RaceSimulator
Simulazione gare con i modelli addestrati: selezione guidata gara/pilota, predizione P(pit) per ogni giro, raccomandazione giro pit ottimale, confronto con strategia reale.

## Quick Start

### Opzione A: Usare modelli pre-addestrati
1. Apri `04_RaceSimulator.ipynb` in Colab
2. Carica i file del modello
3. Seleziona anno, gara, pilota
4. Visualizza risultati

### Opzione B: Training da zero
Esegui i notebook in ordine: DataLoader → DataAnalysis → LSTMModel → RaceSimulator

## Limitazioni Note

- Compound prediction: Accuracy ~78%, tende a preferire compound comuni
- Safety Car: Il modello genera falsi positivi durante SC
- Ultimo stint: FP inevitabili perché il modello non sa che è l'ultimo stint
- Stint corti: Con meno di 5 giri di dati, le predizioni sono meno affidabili
- Condizioni wet: Pochi dati di training per gare bagnate

## Riferimenti

- [FastF1 Documentation](https://docs.fastf1.dev/)
- [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/)

## Disclaimer

I marchi Formula 1, F1, FIA Formula One World Championship, Grand Prix e relativi sono proprietà di Formula One Licensing BV. Questo progetto NON è affiliato, approvato o connesso a Formula One Management, FIA, o alcun team/pilota F1.

# F1 Strategy Predictor

Predizione pit stop e scelta compound per gare di Formula 1 usando modelli LSTM.

---

## Panoramica

Il progetto utilizza dati telemetrici F1 (via FastF1) per addestrare due modelli di deep learning:

1. **Pit Timing Model** - Prevede quando un pilota effettuerà un pit stop (classificazione binaria)
2. **Compound Model** - Prevede quale gomma verrà montata (classificazione multiclass)

I modelli sono basati su architettura LSTM con attention mechanism, addestrati su dati storici dal 2023 al 2025.

---

## Requisiti

- Python 3.10+
- Google Colab (consigliato) o ambiente locale con GPU
- Librerie: TensorFlow, FastF1, pandas, numpy, matplotlib, scikit-learn, joblib

---

## Struttura Progetto

```
F1StrategyPredictor/
├── 01_DataLoader.ipynb      # Download dati da FastF1 API
├── 02_DataAnalysis.ipynb    # EDA e pulizia dati
├── 03_LSTMModel.ipynb       # Training modelli
├── 04_RaceSimulator.ipynb   # Simulazione gare
│
├── Data/
│   ├── f1_dataset_combined.pkl   # Dati grezzi (da DataLoader)
│   ├── f1_dataset_clean.pkl      # Dati puliti (da DataAnalysis)
│   └── f1_dataset_featured.pkl   # Dati con features (da LSTMModel)
│
└── Model/
    ├── f1_pit_model.keras        # Modello pit timing
    ├── f1_compound_model.keras   # Modello compound
    ├── f1_pit_scaler.pkl         # Scaler features pit
    ├── f1_comp_scaler.pkl        # Scaler features compound
    ├── label_encoder.pkl         # Encoder compound labels
    └── modelConfig.json          # Configurazione e metriche
```

---

## Pipeline

### 1. DataLoader

Scarica i dati telemetrici dalla API FastF1.

**Input:** Nessuno (scarica da API)  
**Output:** `f1_dataset_combined.pkl`

**Note importanti:**
- Il download è incrementale (un anno alla volta)
- FastF1 ha rate limit (~100 richieste/ora)
- Disconnetti il runtime tra un anno e l'altro per evitare ban IP
- La cache locale velocizza le esecuzioni successive

**Configurazione:**
```python
YEAR_TO_LOAD = 2024    # Anno da scaricare
MAX_ROUNDS = None      # None = tutte le gare
FORCE_RELOAD = False   # True = riscarica anche se presente
```

---

### 2. DataAnalysis

Analisi esplorativa e pulizia dei dati.

**Input:** `f1_dataset_combined.pkl`  
**Output:** `f1_dataset_clean.pkl`

**Operazioni:**
- Rimozione outlier (tempi giro anomali)
- Gestione valori mancanti
- Analisi distribuzione compound
- Analisi pattern pit stop
- Statistiche per circuito/pilota/team

---

### 3. LSTMModel

Training dei modelli LSTM.

**Input:** `f1_dataset_clean.pkl`  
**Output:** Modelli, scaler, encoder, config

**Architettura:**
- Bidirectional LSTM (64 unita)
- Multi-Head Attention (4 heads)
- Dropout + BatchNormalization
- Sequence length: 10 giri

**Metriche target:**
| Modello | Metrica | Target | Ottenuto |
|---------|---------|--------|----------|
| Pit | AUC-ROC | >= 0.90 | 0.94 |
| Pit | F1 Score | >= 0.70 | 0.75 |
| Compound | Accuracy | >= 0.75 | 0.78 |

**Features principali (pit):**
- LapsRemainingNorm, LapsRemainingPct
- TyreMargin, TyreAgeCubeRatio
- MyStintVsField, PitUrgency
- DeltaFromBest, GapTrend

**Features principali (compound):**
- IsMedium, IsSoft, IsHard
- LapsRemainingPct, StintNum
- TrackTempNorm, IsRaining
- HasUsedSoft/Medium/Hard

---

### 4. RaceSimulator

Simulazione gare con i modelli addestrati.

**Input:** Modelli + `f1_dataset_featured.pkl`  
**Output:** Predizioni, grafici, CSV

**Funzionalita:**
- Selezione guidata gara/pilota
- Predizione P(pit) per ogni giro
- Raccomandazione giro pit ottimale
- Predizione compound
- Confronto con strategia reale
- Export risultati

**Configurazione:**
```python
ANNO = 2024
ROUND = 1        # Numero gara (vedi lista disponibile)
PILOTA = 'VER'   # Codice pilota (vedi lista disponibile)
```

**Output files:**
- `{anno}_R{round}_{pilota}_simulation.csv` - Dati completi
- `{anno}_R{round}_{pilota}_summary.csv` - Riepilogo pit
- `{anno}_R{round}_{pilota}_chart.png` - Grafico

---

## Quick Start

### Opzione A: Usare modelli pre-addestrati

Se hai gia i file del modello:

1. Apri `04_RaceSimulator.ipynb` in Colab
2. Carica i file nella sessione:
   - `f1_pit_model.keras`
   - `f1_compound_model.keras`
   - `f1_pit_scaler.pkl`
   - `f1_comp_scaler.pkl`
   - `label_encoder.pkl`
   - `modelConfig.json`
   - `f1_dataset_featured.pkl`
3. Esegui le celle
4. Seleziona anno, gara, pilota
5. Visualizza risultati

### Opzione B: Training da zero

1. **DataLoader**: Scarica dati 
2. **DataAnalysis**: Pulisci dati 
3. **LSTMModel**: Addestra modelli 
4. **RaceSimulator**: Simula gare

---

## Limitazioni note

1. **Compound prediction**: Accuracy ~78%, tende a preferire compound comuni (Medium/Hard)
2. **Safety Car**: Il modello genera falsi positivi durante SC (normale: opportunita pit)
3. **Ultimo stint**: FP inevitabili perche il modello non sa che e l'ultimo stint
4. **Stint corti**: Con meno di 5 giri di dati, le predizioni sono meno affidabili
5. **Condizioni wet**: Pochi dati di training per gare bagnate

---

## Estensioni possibili

- Aggiungere dati telemetria (velocita settori, temperature freni)
- Integrare previsioni meteo real-time
- Multi-driver prediction per strategia di gara
- Ottimizzazione threshold per circuito
- Deploy come API per uso real-time

---

## Riferimenti

- [FastF1 Documentation](https://docs.fastf1.dev/)
- [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/)
- Dati: Formula 1 timing data via FastF1 (2023-2025)

---

## Dati e Licenza

### Fonte dati

I dati sono ottenuti tramite [FastF1](https://github.com/theOehrly/Fast-F1), libreria open-source (MIT License) che accede a timing data pubblicamente disponibili durante le trasmissioni live F1. FastF1 non utilizza API ufficiali F1.

### Uso accademico

Questo progetto rientra nell'uso educativo come definito nelle [F1 Guidelines](https://www.formula1.com/en/information/guidelines):

> "Limited use of Other Intellectual Property Rights for educational purposes may be acceptable where the use is justified, limited, and non-commercial."

Il progetto:
- E' una tesi universitaria (non commerciale)
- Non ridistribuisce dati grezzi
- Produce solo analisi aggregate e modelli ML
- Non usa loghi, immagini o contenuti brandizzati F1

### Disclaimer

I marchi Formula 1, F1, FIA Formula One World Championship, Grand Prix e relativi sono proprieta di Formula One Licensing BV.

Questo progetto NON e affiliato, approvato o connesso a Formula One Management, FIA, o alcun team/pilota F1.

Vedi file `LICENSE` per dettagli completi.

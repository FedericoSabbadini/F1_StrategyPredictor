from io import BytesIO
import json
import os
import numpy as np
import pandas as pd
import requests
import joblib
import tempfile
from tensorflow.keras.models import load_model


class Race:

    def __init__(
        self,
        url: str = "https://raw.githubusercontent.com/FedericoSabbadini/f1-strategy-predictor/main/03_model/LSTM/output/Model/",
    ):
        # Store models and preprocessing tools
        self.model_pit = self.load_keras_from_url(url + "f1_pit_model.keras")
        self.model_comp = self.load_keras_from_url(url + "f1_compound_model.keras")
        self.scaler_pit = joblib.load(BytesIO(self.download_bytes(url + "f1_pit_scaler.pkl")))
        self.scaler_comp = joblib.load(BytesIO(self.download_bytes(url + "f1_comp_scaler.pkl")))
        self.label_encoder = joblib.load(BytesIO(self.download_bytes(url + "label_encoder.pkl")))

        config = json.loads(self.download_bytes(url + "modelConfig.json"))
        # Model configuration
        self.features = config["FEATURES"]
        self.seq_len = config["sequence_length"]
        self.pit_thresholdAdd = 0.21
        self.pit_threshold = config["pit_threshold"] + self.pit_thresholdAdd
        self.df_f1 = self.get_f1_dataset(url)


    # Sends an HTTP GET request
    def download_bytes(self, url: str) -> bytes:
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    # TF load_model using download_bytes
    def load_keras_from_url(self, url: str):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            f.write(self.download_bytes(url))
            path = f.name
        model = load_model(path)
        os.remove(path)
        return model
    
    def get_f1_dataset(self, base_url: str) -> pd.DataFrame:
        url = base_url + "f1_dataset_featured.pkl"
        df = pd.read_pickle(url)
        return df
from io import BytesIO
import json
import os
import numpy as np
import pandas as pd
from modelLoad import Race

class RaceSimulator:
    """
    Full lap-by-lap race simulator.
    """


    def __init__(
        self,
        race: Race,
    ):
        # Store models and preprocessing tools
        self.model_pit = race.model_pit
        self.model_comp = race.model_comp
        self.scaler_pit = race.scaler_pit
        self.scaler_comp = race.scaler_comp
        self.label_encoder = race.label_encoder

        # Model configuration
        self.features = race.features
        self.seq_len = race.seq_len
        self.pit_threshold = race.pit_threshold
        self.pit_thresholdAdd = race.pit_thresholdAdd


    def _make_sequence(self, history_df, scaler):
        """
        Turn past laps into an LSTM input tensor.
        Shape: (1, seq_len, n_features)
        """

        # Giro 0, no past data = fully zero sequence (remember padding/masking)
        if len(history_df) == 0:
            return np.zeros((1, self.seq_len, len(self.features)), dtype=np.float32)

        # Extract and scale feature values
        X = history_df[self.features].values.astype(np.float32)
        X = scaler.transform(X)

        # Left-pad with 0 if too short
        if len(X) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X), X.shape[1]), dtype=np.float32)
            X = np.vstack([pad, X])
        else:
            # Keep only last seq_len's laps
            X = X[-self.seq_len:]

        # Add batch dimension
        return X.reshape(1, self.seq_len, X.shape[1])


    def simulate(self, race_df, driver):
        """
        Simulate the race lap by lap for one driver.
        """

        # Filter and sort driver laps
        df = (
            race_df[race_df["Driver"] == driver]
            .sort_values("LapNumber")
            .reset_index(drop=True)
        )

        total_laps = int(df["LapNumber"].max())
        num_stints = int(df["Stint"].max())

        # Start of a new stint (lap)
        pit_laps = set()
        pit_laps_window = set()
        for s in range(2, num_stints + 1):
            first_lap = int(df[df["Stint"] == s - 1]["LapNumber"].max())
            pit_laps.add(first_lap)
            pit_laps_window.add(first_lap-1)


        # Safety-car laps
        sc_laps = set(
            df[df.get("UnderCaution", 0) == 1]["LapNumber"].astype(int)
        )

        results = []

        # Process lap by lap
        for i in range(len(df)):
            row = df.iloc[i]
            lap = int(row["LapNumber"])
            stint = int(row["Stint"])

            # History = all previous laps in this stint
            history = df[(df["Stint"] == stint) & (df["LapNumber"] < lap)]

            # --------------------
            # PIT PROBABILITY
            # --------------------
            seq = self._make_sequence(history, self.scaler_pit)
            # Predict se Pit in 3 Laps or Not
            pit_prob = float(self.model_pit.predict(seq, verbose=0).ravel()[0])
            pit_pred = int(pit_prob >= self.pit_threshold) # Threshold applied

            # --------------------
            # COMPOUND PREDICTION
            # --------------------
            comp_pred = None
            comp_actual = None

            # Only predict compound if this lap is a pit lap, to make easy the comparison
            # Predict compound
            prev_stint_df = df[df["LapNumber"] < lap]
            seq = self._make_sequence(prev_stint_df, self.scaler_comp)

            probs = self.model_comp.predict(seq, verbose=0)[0]
            comp_pred = self.label_encoder.inverse_transform(
                [probs.argmax()]
            )[0]

            comp_actual = row["Compound"]

            # --------------------
            # STORE RESULT
            # --------------------
            results.append({
                "Lap": lap,
                "TotalLaps": total_laps,
                "Stint": stint,
                "Compound": row["Compound"],
                "TyreAge": int(row.get("TyreLife", 0)),
                "PitProb": pit_prob,
                "PitPred": pit_pred,
                "PitThisLap": int(lap in pit_laps_window),
                "IsLastStint": stint == num_stints,
                "CompPred": comp_pred,
                "CompActual": comp_actual,
                "UnderCaution": int(row.get("UnderCaution", 0)),
                "DataAvailable": True,
                "MissingReason": None
            })

        result_df = pd.DataFrame(results)

        # --------------------
        # RECOMMENDED PIT LAPS
        # --------------------
        recommended = {}
        THRS = self.pit_threshold
        for s in range(1, num_stints):
            rows = result_df[
                (result_df["Stint"] == s)
            ]
            rows_probMax = rows[rows['PitProb']>THRS].copy()
            rows_probNotMax = rows[rows['PitProb']>THRS-(self.pit_thresholdAdd/2)].copy()
            if len(rows_probMax) > 0:
                idx = rows_probMax["PitProb"].idxmin()
                recommended[s] = {
                    "lap": int(rows_probMax.loc[idx, "Lap"]),
                    "prob": float(rows_probMax.loc[idx, "PitProb"])
                }
            elif len(rows_probNotMax) > 0:
                THRS = THRS - (self.pit_thresholdAdd/2)
                idx = rows_probNotMax["PitProb"].idxmin()
                recommended[s] = {
                    "lap": int(rows_probNotMax.loc[idx, "Lap"]),
                    "prob": float(rows_probNotMax.loc[idx, "PitProb"])
                }
            else:
               prob = rows['PitProb'].max()
               rows_prob = rows[rows['PitProb']==prob].copy()
               THRS = THRS - self.pit_thresholdAdd
               if len(rows_prob) > 0 and prob>=self.pit_threshold-self.pit_thresholdAdd:
                idx = rows_prob["PitProb"].idxmin()
                recommended[s] = {
                    "lap": int(rows_prob.loc[idx, "Lap"]),
                    "prob": float(rows_prob.loc[idx, "PitProb"])
                }


        return result_df, pit_laps, sc_laps, recommended, num_stints, THRS
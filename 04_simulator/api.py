# server.py — FastAPI corretta per ModelVault
# Lancia con: uvicorn server:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from RaceSimulator import simulate
import base64, io

app = FastAPI()

# CORS — necessario per le chiamate da modelvault.it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.modelvault.it", "https://modelvault.it", "http://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/simulate")
def run_simulation(year: int, round: int, driver: str):
    """
    Chiamato da modelvault.it/console/panel.php
    Params: year (2022-2025), round (1-23), driver (3 lettere, es. VER)
    Returns: { simulation: csv, pit_summary: csv, chart: base64_png }
    """
    simulation_csv, summary_csv, chart_base64 = simulate(year, round, driver)
    return {
        "simulation": simulation_csv,
        "pit_summary": summary_csv,
        "chart": chart_base64
    }

# Endpoint compatibile con proxy.php (POST)
@app.post("/simulate")
def run_simulation_post(data: dict = {}):
    form = data.get("form_data", data)
    year = int(form.get("year", 2024))
    rnd = int(form.get("round", 1))
    driver = str(form.get("driver", "VER")).upper()
    simulation_csv, summary_csv, chart_base64 = simulate(year, rnd, driver)
    return {
        "reply": "Simulazione completata.",
        "simulation": simulation_csv,
        "pit_summary": summary_csv,
        "chart": chart_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

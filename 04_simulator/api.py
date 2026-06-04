from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from RaceSimulator import simulate
import base64, io
import uvicorn

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
) # Permette richieste da qualsiasi origine 
# (utile per sviluppo, ma da restringere in produzione)

@app.get("/simulate")
def run_simulation(year: int, round: int, driver: str):
    """
    Chiamato da modelvault.it/console/panel.php
    con la query: ?year=2024&round=1&driver=VER
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

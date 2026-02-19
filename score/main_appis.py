import pandas as pd
import os

from modules.one_stability import train_stability_model, predict_stability
from modules.two_fatigue import train_fatigue_model, predict_fatigue
from modules.three_injury import train_injury_model, predict_injury


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "appis_simulated.csv")

df = pd.read_csv(CSV_PATH)


# MODULE 1

stability_model, stability_scaler = train_stability_model(df)
stability_scores = predict_stability(stability_model, stability_scaler, df)

# MODULE 2

fatigue_model, fatigue_scaler = train_fatigue_model(df)
fatigue_scores = predict_fatigue(fatigue_model, fatigue_scaler, df)


# MODULE 3

injury_model, injury_scaler = train_injury_model(df, stability_scores)
injury_scores = predict_injury(injury_model, injury_scaler, df, stability_scores)


# READINESS

min_len = min(len(stability_scores), len(fatigue_scores), len(injury_scores))

stability_scores = stability_scores[:min_len]
fatigue_scores = fatigue_scores[:min_len]
injury_scores = injury_scores[:min_len]

readiness = (
    0.45 * stability_scores +
    0.35 * (100 - fatigue_scores) +
    0.20 * (100 - injury_scores)
)


print("\nAPPIS Results\n")
print("Stability:", stability_scores[:5])
print("Fatigue:", fatigue_scores[:5])
print("Injury:", injury_scores[:5])
print("Readiness:", readiness[:5])



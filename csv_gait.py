# appis_clean_demo_csv.py

"""
APPIS - Adaptive Prosthetic Performance Intelligence System
CSV-Based Demo Version with Streaming Output and Final Plot
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest  # type: ignore
import time
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# 1. LOAD CSV DATA
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "appis_simulated.csv")

df = pd.read_csv(CSV_FILE)

# -------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------
ROLLING_WINDOW = 3

# Symmetry ratio (can use CSV column or recompute)
df['symmetry_ratio'] = df['prosthetic_load'] / df['intact_load']

# Use CSV's knee_angle_variance directly and compute rolling averages
df['socket_pressure_avg'] = df['socket_pressure'].rolling(window=ROLLING_WINDOW).mean()
df['angular_velocity_avg'] = df['angular_velocity'].rolling(window=ROLLING_WINDOW).mean()
df['ground_contact_avg'] = df['ground_contact_time'].rolling(window=ROLLING_WINDOW).mean()
df['stride_length_avg'] = df['stride_length'].rolling(window=ROLLING_WINDOW).mean()

df = df.dropna().reset_index(drop=True)

features = [
    'symmetry_ratio',
    'socket_pressure_avg',
    'angular_velocity_avg',
    'ground_contact_avg',
    'stride_length_avg',
    'knee_angle_variance'
]

X = df[features]

# -------------------------------------------------
# 3. MODULE 1: STABILITY MODEL
# -------------------------------------------------
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

raw_anomalies = -model.decision_function(X)
mean_a = np.mean(raw_anomalies)
normalized_anomalies = 1 / (1 + np.exp(-5 * (raw_anomalies - mean_a)))

# Exponential smoothing
alpha = 0.3
smoothed = [normalized_anomalies[0]]
for val in normalized_anomalies[1:]:
    smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
normalized_anomalies = np.array(smoothed)
stability_scores = 100 * (1 - normalized_anomalies)

# -------------------------------------------------
# 4. FATIGUE MODEL (Nonlinear Demo Version)
# -------------------------------------------------
baseline_stride = df['stride_length_avg'].iloc[0]

def compute_fatigue(row):
    hr_component = (row['heart_rate'] - 100) * 0.6
    gct_component = (row['ground_contact_avg'] - 0.65) * 200
    stride_component = (baseline_stride - row['stride_length_avg']) * 150
    fatigue = hr_component + gct_component + stride_component
    return np.clip(fatigue, 0, 100)

# -------------------------------------------------
# 5. INJURY RISK MODEL
# -------------------------------------------------
def compute_injury(row, anomaly):
    symmetry_risk = abs(1 - row['symmetry_ratio']) * 100
    pressure_risk = (row['socket_pressure_avg'] - 30) * 2
    anomaly_risk = anomaly * 100
    risk = 0.4 * symmetry_risk + 0.3 * pressure_risk + 0.3 * anomaly_risk
    return np.clip(risk, 0, 100)

# -------------------------------------------------
# 6. READINESS SCORE
# -------------------------------------------------
def compute_readiness(stability, fatigue, injury):
    return 0.45 * stability + 0.35 * (100 - fatigue) + 0.20 * (100 - injury)

# -------------------------------------------------
# 7. COMPUTE ALL METRICS
# -------------------------------------------------
fatigue_scores = [compute_fatigue(r) for _, r in df.iterrows()]
injury_scores = [compute_injury(row, anomaly) for (_, row), anomaly in zip(df.iterrows(), normalized_anomalies)]

readiness_scores = [compute_readiness(s, f, i) for s, f, i in zip(stability_scores, fatigue_scores, injury_scores)]

# -------------------------------------------------
# 8. STREAMING OUTPUT
# -------------------------------------------------
print("\nAPPIS - Adaptive Prosthetic Intelligence (CSV Mode)\n")
print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Time", "Stability", "Fatigue", "Injury", "Readiness"))

for i in range(len(df)):
    print("{:<10} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f}".format(
        str(df['timestamp'][i]),
        stability_scores[i],
        fatigue_scores[i],
        injury_scores[i],
        readiness_scores[i]
    ))
    time.sleep(0.5)

# -------------------------------------------------
# 9. FINAL PLOT
# -------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(df['timestamp'], stability_scores, label='Stability', marker='o')
plt.plot(df['timestamp'], fatigue_scores, label='Fatigue', marker='x')
plt.plot(df['timestamp'], injury_scores, label='Injury', marker='s')
plt.plot(df['timestamp'], readiness_scores, label='Readiness', marker='d')
plt.xticks(rotation=45)
plt.ylabel('Score')
plt.xlabel('Timestamp')
plt.title('APPIS Metrics Over Time')
plt.legend()
plt.tight_layout()
plt.show()

print("\nDemo Complete.")


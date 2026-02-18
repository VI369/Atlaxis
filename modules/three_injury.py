import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def train_injury_model(df, stability_scores):

    df = df.copy()

    df['symmetry_ratio'] = df['prosthetic_load'] / df['intact_load']
    df['symmetry_change'] = df['symmetry_ratio'].diff().fillna(0)
    df['intact_load_spike'] = df['intact_load'].diff().fillna(0)
    df['socket_pressure_level'] = df['socket_pressure']
    df['baseline_anomaly'] = 100 - stability_scores

    df = df.dropna().reset_index(drop=True)

    features = [
        'symmetry_ratio',
        'symmetry_change',
        'intact_load_spike',
        'socket_pressure_level',
        'baseline_anomaly'
    ]

    X = df[features]

    # Binary injury label (example threshold logic for demo)
    y = (df['socket_pressure'] > df['socket_pressure'].mean()).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X_scaled, y)

    return model, scaler


def predict_injury(model, scaler, df, stability_scores):

    df = df.copy()

    df['symmetry_ratio'] = df['prosthetic_load'] / df['intact_load']
    df['symmetry_change'] = df['symmetry_ratio'].diff().fillna(0)
    df['intact_load_spike'] = df['intact_load'].diff().fillna(0)
    df['socket_pressure_level'] = df['socket_pressure']
    df['baseline_anomaly'] = 100 - stability_scores

    df = df.dropna().reset_index(drop=True)

    features = [
        'symmetry_ratio',
        'symmetry_change',
        'intact_load_spike',
        'socket_pressure_level',
        'baseline_anomaly'
    ]

    X = df[features]
    X_scaled = scaler.transform(X)

    injury_prob = model.predict_proba(X_scaled)[:, 1]

    return injury_prob * 100


# module2_fatigue.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore


def train_fatigue_model(df):

    df['symmetry_ratio'] = df['prosthetic_load'] / df['intact_load']

    df['gct_mean'] = df['ground_contact_time'].rolling(5).mean()
    df['stride_std'] = df['stride_length'].rolling(5).std()
    df['symmetry_trend'] = df['symmetry_ratio'].rolling(5).mean()
    df['hr_trend'] = df['heart_rate'].rolling(5).mean()
    df['pressure_trend'] = df['socket_pressure'].rolling(5).mean()

    df = df.dropna().reset_index(drop=True)

    features = [
        'gct_mean',
        'stride_std',
        'symmetry_trend',
        'hr_trend',
        'pressure_trend'
    ]

    X = df[features]
    y = df['fatigue_label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X_scaled, y)

    return model, scaler


def predict_fatigue(model, scaler, df):

    features = [
        'gct_mean',
        'stride_std',
        'symmetry_trend',
        'hr_trend',
        'pressure_trend'
    ]

    X = df[features]
    X_scaled = scaler.transform(X)

    fatigue_scores = model.predict(X_scaled)

    return np.clip(fatigue_scores, 0, 100)

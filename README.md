# Atlaxis - An adaptive Prosthetic Performance Intelligence System

> A cloud-based machine learning system that models prosthetic biomechanics to predict fatigue progression, neuromuscular instability, and injury risk — generating a unified **Readiness Index (0–100)** for para-athletes.

---

## Overview

**Atlaxis** is an intelligent biomechanical analytics platform designed for smart lower-limb prosthetics.

Instead of relying on fixed rule-based thresholds, Atlaxis builds a **personalized biomechanical distribution model** for each athlete and continuously learns:

* Normal asymmetric gait patterns
* Fatigue progression dynamics
* Compensatory overload behavior
* Injury risk escalation signals

The system processes simulated prosthetic sensor data (CSV time-series) and outputs a real-time:

```
Readiness Index (0–100)
```

Example:

```
Readiness Index: 68 / 100
Status: Moderate Risk
```

---

## Core Innovation

Unlike generic monitoring systems, Atlaxis:

* Models **personal asymmetry** rather than comparing against global norms
* Learns **nonlinear fatigue progression**
* Detects **compensatory overload before injury**
* Continuously adapts after each session

This makes it suitable for para-athletes with stable but asymmetric gait signatures.

---

## Simulated Prosthetic Data (CSV Schema)

The system assumes a smart lower-limb prosthetic streaming:

| Column              | Description                     |
| ------------------- | ------------------------------- |
| timestamp           | Time-series index               |
| prosthetic_load     | Load on prosthetic limb         |
| intact_load         | Load on intact limb             |
| socket_pressure     | Residual limb socket pressure   |
| knee_angle          | Knee flexion angle              |
| angular_velocity    | Angular joint velocity          |
| ground_contact_time | Stance duration                 |
| step_time           | Step timing                     |
| stride_length       | Distance per stride             |
| heart_rate          | Physiological fatigue indicator |
| fatigue_label       | Supervised training label       |

All values are numeric and time-series based.

---

# System Architecture

## 1️⃣ Personalized Gait Baseline Model

**Objective:**
Model each athlete’s normal biomechanical distribution.

### Method

* Gaussian Mixture Model (GMM)
  OR
* Isolation Forest

### Feature Vector (Per Window)

```
X = [
  symmetry_ratio,
  socket_pressure,
  angular_velocity,
  ground_contact_time,
  stride_length,
  knee_angle_variance
]
```

Where:

```
symmetry_ratio = prosthetic_load / intact_load
```

### Output

Anomaly Score (0–1)

High score → deviation from personal baseline.

---

## 2️⃣ Fatigue Progression Predictor

Fatigue evolves temporally and nonlinearly.

### Observed Patterns

* ↑ Ground contact time
* ↓ Stride length
* ↑ Variability
* ↑ Heart rate

### Model

* XGBoost Regressor
  OR
* Random Forest Regressor

### Input Features (Rolling Windows)

* mean_ground_contact_time
* std_stride_length
* symmetry_trend
* heart_rate_trend
* socket_pressure_trend

### Output

```
Predicted Fatigue Score (0–100)
```

Tree-based models are chosen because:

* They handle nonlinear biomechanics
* Perform well with small datasets
* Provide feature importance interpretability

---

## 3️⃣ Injury Risk Classifier

In para-athletes, injury risk primarily arises from:

* Increasing asymmetry
* Intact limb overload
* Rising socket pressure

### Model

* XGBoost Classifier
  OR
* Random Forest Classifier

### Features

* current_symmetry
* symmetry_rate_of_change
* intact_load_spike
* socket_pressure_level
* baseline_anomaly_score

### Target

```
0 = Safe
1 = Elevated Risk
```

---

## 4️⃣ Optional Temporal Modeling (Advanced Layer)

For stronger progression modeling:

### Model

Lightweight LSTM

### Input

Last 20 time windows

### Output

Fatigue probability in next interval

This captures dynamic progression patterns beyond static windows.

---

# Unified Readiness Score

All model outputs are normalized:

* Fatigue Score (0–100)
* Stability Score (derived from anomaly score)
* Injury Risk Probability

### Weighted Formula (Para-Specific)

Symmetry and injury carry higher weight.

```
Readiness =
(0.45 × Stability Score)
+ (0.35 × (100 − Fatigue Score))
+ (0.20 × (100 − Injury Risk Probability))
```

### Final Output Example

```
Readiness Index = 68 / 100
Status = Moderate Risk
```

---

# Tech stack used for implementation

* Python
* Pandas (CSV simulation)
* NumPy
* Scikit-learn
* XGBoost
* Streamlit (Dashboard UI)

No physical hardware required. (Future integration)
Simulated CSV behaves like a prosthetic data stream.

---

# Project Structure

```
Atlaxis/
│
├── dashboard/                  
│   └── appis_full.py
│
├── data/                       
│   ├── appis_simulated.csv
│
├── models/                     
│   ├── one_stability.py
│   ├── two_fatigue.py
│   └── three_injury.py
│
├── score/                        
│   ├── main_appis.py
│
├── requirements.txt
├── README.md
└── LICENSE

```

---

# Installation

```bash
git clone https://github.com/yourusername/Atlaxis.git
cd atlaxis
pip install -r requirements.txt
streamlit run appis_full.py
```

---

# What makes Atlaxis different than most Threshold Systems

* Personalized distribution modeling for robust analysis of an individual players' performance
* Nonlinear fatigue learning 
* Time-aware risk detection for timely injury prevention
* Interpretable tree-based ML
* No need of visual information — pure biomechanical intelligence based on the prosthetic data stream

---

# Future Roadmap

* Real-time IoT prosthetic integration
* Cloud model deployment (AWS/GCP)
* Athlete-specific transfer learning
* Reinforcement learning for adaptive tuning
* Mobile dashboard integration

---

# License

Creative Commons Zero v1.0 Universal

---

# Author

Team Aetherion 
:)

---

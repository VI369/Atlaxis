import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


st.set_page_config(page_title="APPIS Mission Control", layout="wide", page_icon="🦾")

import ollama

def get_local_ai_insight(metrics_dict, importance_dict):
    prompt = f"""
    You are a prosthetics performance expert.

    Athlete Metrics:
    {metrics_dict}

    Top Risk Factors:
    {importance_dict}

    Give a short, 2-sentence actionable recommendation.
    """

    try:
        response = ollama.chat(
            model='phi3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception:
        return "Local AI is offline. Run: ollama serve"



# MOCK DATA (OR UPLOAD CSV)

@st.cache_data
def get_mock_data():
    return pd.DataFrame({
        "prosthetic_load": np.random.normal(20, 5, 500),
        "intact_load": np.random.normal(85, 7, 500),
        "symmetry_ratio": np.random.normal(0.4, 0.05, 500),
        "heart_rate": np.random.normal(160, 10, 500),
        "socket_pressure": np.random.normal(20, 3, 500),
        "ground_contact_time": np.random.normal(0.6, 0.05, 500),
        "stride_length": np.random.normal(1.0, 0.1, 500)
    })

# SIDEBAR

with st.sidebar:
    st.title("⚙️ APPIS Control Panel")
    uploaded_file = st.file_uploader("Upload Session CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = get_mock_data()

    mode = st.radio("Mode", ["Athlete Mode", "Coach Mode"])

   
    
with st.sidebar:

    st.markdown("""
        <h2 style='text-align: center; color: #ff4b4b;'> ATLAXIS LAB</h2>
        <hr style="margin-top: 0px;">
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Patient Files",
            "Gait Analysis",
            "Reports",
            "Settings"
        ],
        icons=[
            "house",
            "folder2-open",
            "activity",
            "file-earmark-bar-graph",
            "gear"
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#0e1117",
            },
            "icon": {
                "color": "#888",
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#262730",
            },
            "nav-link-selected": {
                "background-color": "#ff4b4b",
                "color": "white",
            },
        },
    )

    st.markdown("---")
    st.info("Patient ID: **GH-2024-88**")



# CORE METRICS

df["asymmetry"] = abs(df["symmetry_ratio"])

readiness_score = 100 - (df["asymmetry"].mean()*100)
readiness_score = np.clip(readiness_score, 0, 100)

stability = round(df["asymmetry"].mean()*100,2)
fatigue = round(df["heart_rate"].mean(),1)
injury_risk = round((df["intact_load"].mean()/df["prosthetic_load"].mean())*10,1)

previous_score = readiness_score - np.random.uniform(-5,5)


# ALERTS

if injury_risk > 65:
    st.error("🚨 Injury probability exceeded safe threshold.")
elif readiness_score < 75:
    st.warning("⚠️ Moderate biomechanical risk detected.")
else:
    st.success("🟢 Stable performance state.")





# HERO SECTION

st.title("Atlaxis — Adaptive Prosthetic Intelligence")

left, center, right = st.columns([1, 2, 1])  # Middle column wider

with center:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=readiness_score,
        delta={'reference': previous_score},
        title={'text': "Unified Readiness"},
        gauge={
            'axis': {'range':[0,100]},
            'steps':[
                {'range':[0,50],'color':'#ff4b4b'},
                {'range':[50,75],'color':'#f9c74f'},
                {'range':[75,100],'color':'#43aa8b'}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)


# MODULE CARDS

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🟦 Stability")
    st.metric("Anomaly Score", f"{stability}%")
    st.line_chart(df["symmetry_ratio"].rolling(20).mean())

with col2:
    st.subheader("🟧 Fatigue")
    st.metric("Heart Rate", f"{fatigue} bpm")
    st.line_chart(df["heart_rate"].rolling(20).mean())

with col3:
    st.subheader("🟥 Injury Risk")
    st.metric("Risk Index", f"{injury_risk}%")
    st.progress(int(injury_risk))


# LIVE MONITORING

st.subheader("Live Biomechanics Stream")

metric = st.selectbox("Select Metric", df.columns)

baseline = df[metric].mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=df[metric], mode="lines", name="Live"))
fig2.add_trace(go.Scatter(y=[baseline]*len(df), mode="lines", name="Baseline"))
st.plotly_chart(fig2, use_container_width=True)


# BASELINE VISUALIZER

st.subheader("Personal Baseline Model")

X = df[["prosthetic_load","intact_load"]].values
gmm = GaussianMixture(n_components=1, random_state=42)
gmm.fit(X)

current_point = np.array([[X[-1][0], 60]])
log_likelihood = gmm.score_samples(current_point)

threshold = np.percentile(gmm.score_samples(X),5)
point_color = "red" if log_likelihood < threshold else "green"

fig3 = go.Figure()
fig3.update_yaxes(
    range=[0, 100],        # keeps full scale
    tickmode='array',
    tickvals=[20, 40, 60, 80]
)
fig3.add_trace(go.Scatter(
    x=X[:,0],
    y=X[:,1],
    mode='markers',
    opacity=0.4,
    name="Past Scores"
))
fig3.add_trace(go.Scatter(x=current_point[:,0], y=current_point[:,1],
                          mode='markers',
                          marker=dict(size=14,color=point_color),
                          name="Current Score"))
st.plotly_chart(fig3, use_container_width=True)


# COACH MODE EXPLAINABILITY

if mode == "Coach Mode":
    st.subheader("Injury Risk Explainability")

    df_model = df.copy()
    df_model["injury"] = np.where(df_model["symmetry_ratio"]>1.05,1,0)

    features = df_model.drop("injury",axis=1)
    target = df_model["injury"]

    model = RandomForestClassifier()
    model.fit(features,target)

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features.columns,
        "Importance": importances
    }).sort_values("Importance",ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))


# SMART RECOMMENDATIONS

st.subheader("Smart Recommendations")

if injury_risk > 65:
    st.error("Session termination recommended.")
elif fatigue > 150:
    st.warning("Reduce intensity by 12% for next 10 minutes.")
elif stability > 10:
    st.warning("Perform gait recalibration drills.")
else:
    st.success("Maintain current performance load.")

import ollama

# ollama integration
# def get_local_ai_insight(metrics_dict, importance_dict):
#     """
#     Talks to Llama 3 running locally via Ollama.
#     """
#     prompt = f"""
#     You are the APPIS Intelligence Coach. 
#     Analyze these prosthetic biomechanics:
#     {metrics_dict}
    
#     Top Risk Factors:
#     {importance_dict}
    
#     Provide a professional, 2-sentence tactical recommendation for the athlete.
#     """
    
#     try:
#         response = ollama.chat(model='llama3', messages=[
#             {'role': 'user', 'content': prompt},
#         ])
#         return response['message']['content']
#     except Exception as e:
#         return "Local AI is offline. Ensure Ollama is running ('ollama serve')."

# # --- UI Integration ---
# st.divider()
# st.subheader("🧠 Local AI Mission Control")

# if st.button("Run Local Deep Analysis"):
#     with st.spinner("Llama 3 is processing biomechanics..."):
#         # Passing your existing metrics and importance data
#         insight = get_local_ai_insight(current_metrics, importance_df.head(3).to_dict())
        
#         # Premium Glow Box
#         st.markdown(f"""
#             <div style="background-color: #10141b; padding: 20px; border: 1px solid #007BFF; 
#                         border-radius: 10px; box-shadow: 0 0 15px rgba(0,123,255,0.2);">
#                 <p style="color: #007BFF; font-weight: bold; margin-bottom: 5px;">🤖 APPIS LOCAL INTELLIGENCE</p>
#                 <p style="color: #E0E6ED; font-style: italic;">"{insight}"</p>
#             </div>
#         """, unsafe_allow_html=True)

# 1. DEFINE the metrics first
current_metrics = {
    "Readiness": f"{readiness_score:.1f}%",
    "Heart Rate": fatigue,
    "Symmetry": stability,
    "Injury Risk": f"{injury_risk}%"
}

# 2. NOW use it in the button logic
st.divider()
st.subheader("Get Detailed Insights")

if st.button("Run Local Deep Analysis"):
    with st.spinner("phi3 is processing biomechanics..."):
        importance_df = None


        if importance_df is not None:
            top_features = importance_df.head(3).to_dict()
        else:
            top_features = {"info": "Coach mode required for feature importance"}

        insight = get_local_ai_insight(current_metrics, top_features)

        st.markdown(f"""
            <div style="background-color: #10141b; padding: 20px; border: 1px solid #007BFF; 
                        border-radius: 10px; box-shadow: 0 0 15px rgba(0,123,255,0.2);">
                <p style="color: #007BFF; font-weight: bold; margin-bottom: 5px;">🤖 APPIS LOCAL INTELLIGENCE</p>
                <p style="color: #E0E6ED; font-style: italic;">"{insight}"</p>
            </div>
        """, unsafe_allow_html=True)


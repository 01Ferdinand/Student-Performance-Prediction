import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODELS & DATA
# -----------------------------
@st.cache_data
def load_features():
    feats = pd.read_csv("final_selected_features.csv")["Selected_Features"].tolist()
    return feats

@st.cache_resource
def load_model():
    model = joblib.load("trained_lasso_model.pkl")          # Change to LASSO or Ridge if needed
    scaler = joblib.load("scaler.pkl")
    return model, scaler

selected_features = load_features()
model, scaler = load_model()


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎓 Student Performance Prediction (G3) with SHAP Explainability")

st.write("""
This app predicts the student's final grade **G3** using the trained model.  
It also provides **SHAP explanations** to interpret how each feature contributes.
""")

st.header("📌 Enter Student Feature Values")

# INPUT BOXES
user_data = {}
for feature in selected_features:
    user_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# Scale input
input_scaled = scaler.transform(input_df)


# -----------------------------
# SHAP helper for JS-based plots
# -----------------------------
def st_shap(plot, height=None):
    """Render SHAP plots in Streamlit."""
    import streamlit.components.v1 as components
    shap_html = f"<html><head>{shap.getjs()}</head><body>{plot.html()}</body></html>"
    components.html(shap_html, height=height)


# -----------------------------
# PREDICTION + SHAP
# -----------------------------
if st.button("Predict Grade (G3)"):
    pred = model.predict(input_scaled)[0]
    st.success(f"🎯 Predicted Final Grade (G3): **{pred:.2f}**")

    st.subheader("📊 SHAP Explanation")

    # Use a proper background dataset for SHAP (required)
    background = np.zeros((1, input_scaled.shape[1]))  # safe fallback

    # SHAP Explainer
    explainer = shap.Explainer(model, background)
    shap_values = explainer(input_scaled)

    # -------------------------
    # FORCE PLOT (JS version)
    # -------------------------
    st.write("### 🔵 SHAP Force Plot")
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        input_df.iloc[0]
    )
    st_shap(force_plot, height=250)

    # -------------------------
    # WATERFALL PLOT
    # -------------------------
    st.write("### 🟣 SHAP Waterfall Plot")
    fig_w = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig_w)
    plt.close(fig_w)

    # -------------------------
    # BAR PLOT
    # -------------------------
    st.write("### 🟠 Feature Impact Bar Plot")
    fig_b = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig_b)
    plt.close(fig_b)

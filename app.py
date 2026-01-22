import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Based NIDS",
    layout="wide"
)

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Final Year Student Project**  
Random Forest–based Network Intrusion Detection  
with **Groq AI–powered explanations**
""")

# --------------------------------------------------
# DATASET CONFIG (RAW GITHUB URL)
# --------------------------------------------------
DATA_FILE = (
    "https://raw.githubusercontent.com/"
    "Anjali-K-S25/Vois/main/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("1. API Configuration")
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Starts with gsk_"
)

st.sidebar.header("2. Model Control")

# --------------------------------------------------
# DATA LOADING (FINAL FIXED VERSION)
# --------------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(
            filepath,
            sep=",",
            engine="python",        # Required for CIC-IDS
            encoding="latin1",
            on_bad_lines="skip",    # Skip malformed rows
            nrows=15000             # Memory safe
        )

        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        return df

    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return None

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
def train_model(df):
    features = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Fwd Packet Length Max",
        "Flow IAT Mean"
    ]

    target = "Label"
    df[target] = df[target].astype(str)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy, features, X_test, y_test

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = load_data(DATA_FILE)

if df is None:
    st.stop()

st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
if st.sidebar.button("Train Model"):
    with st.spinner("Training Random Forest model..."):
        model, acc, features, X_test, y_test = train_model(df)

        st.session_state["model"] = model
        st.session_state["features"] = features
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.sidebar.success(f"Model Trained — Accuracy: {acc:.2%}")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
st.header("3. Threat Analysis Dashboard")

if "model" in st.session_state:
    col1, col2 = st.columns(2)

    # ------------------------------
    # Traffic Simulation
    # ------------------------------
    with col1:
        st.subheader("Traffic Simulation")
        if st.button("🎲 Capture Random Network Flow"):
            idx = np.random.randint(len(st.session_state["X_test"]))
            st.session_state["packet"] = st.session_state["X_test"].iloc[idx]
            st.session_state["true_label"] = st.session_state["y_test"].iloc[idx]

    # ------------------------------
    # Analysis Section
    # ------------------------------
    if "packet" in st.session_state:
        packet = st.session_state["packet"]

        with col1:
            st.subheader("Captured Flow Features")
            st.dataframe(packet.to_frame("Value"), use_container_width=True)

        with col2:
            st.subheader("Detection Result")

            packet_df = pd.DataFrame(
                [packet],
                columns=st.session_state["features"]
            )

            prediction = st.session_state["model"].predict(packet_df)[0]

            if prediction == "BENIGN":
                st.success("STATUS: SAFE (BENIGN TRAFFIC)")
            else:
                st.error(f"STATUS: ATTACK DETECTED ({prediction})")

            st.caption(f"Ground Truth Label: {st.session_state['true_label']}")

            st.markdown("---")
            st.subheader("Ask AI Security Analyst (Groq)")

            if st.button("Generate AI Explanation"):
                if not groq_api_key:
                    st.warning("Please enter your Groq API key in the sidebar.")
                else:
                    client = Groq(api_key=groq_api_key)

                    prompt = f"""
You are a cybersecurity analyst.

Prediction: {prediction}

Network flow feature values:
{packet.to_string()}

Explain clearly for a student:
1. Why this traffic was classified as {prediction}
2. What these feature values indicate
3. Whether this flow looks normal or suspicious
"""

                    with st.spinner("Groq AI is analyzing traffic..."):
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.5
                        )

                        st.info(response.choices[0].message.content)

else:
    st.info("Click **Train Model** from the sidebar to begin detection.")
sidebar to begin detection.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Based NIDS",
    layout="wide"
)

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**  
Random Forest–based Network Intrusion Detection with **Groq AI explanations**
""")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_FILE = (
    "https://raw.githubusercontent.com/"
    "Anjali-K-S25/Vois/main/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Starts with gsk_"
)

st.sidebar.header("2. Model Training")

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(
            filepath,
            nrows=15000,
            encoding="latin1"
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
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Fwd Packet Length Max',
        'Flow IAT Mean'
    ]

    target = 'Label'
    df[target] = df[target].astype(str)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
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
# TRAIN BUTTON
# --------------------------------------------------
if st.sidebar.button("Train Model Now"):
    with st.spinner("Training Random Forest model..."):
        model, acc, features, X_test, y_test = train_model(df)
        st.session_state.update({
            "model": model,
            "features": features,
            "X_test": X_test,
            "y_test": y_test
        })
        st.sidebar.success(f"Training Complete — Accuracy: {acc:.2%}")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
st.header("3. Threat Analysis Dashboard")

if "model" in st.session_state:
    col1, col2 = st.columns(2)

    # ---- Packet Selection ----
    with col1:
        st.subheader("Traffic Simulation")
        if st.button("🎲 Capture Random Packet"):
            idx = np.random.randint(len(st.session_state["X_test"]))
            st.session_state["packet"] = st.session_state["X_test"].iloc[idx]
            st.session_state["true_label"] = st.session_state["y_test"].iloc[idx]

    # ---- Packet Analysis ----
    if "packet" in st.session_state:
        packet = st.session_state["packet"]

        with col1:
            st.subheader("Packet Features")
            st.dataframe(packet.to_frame("Value"))

        with col2:
            st.subheader("Detection Result")

            packet_df = pd.DataFrame(
                [packet],
                columns=st.session_state["features"]
            )

            prediction = st.session_state["model"].predict(packet_df)[0]

            if prediction == "BENIGN":
                st.success("STATUS: SAFE (BENIGN)")
            else:
                st.error(f"STATUS: ATTACK DETECTED ({prediction})")

            st.caption(f"Ground Truth: {st.session_state['true_label']}")

            st.markdown("---")
            st.subheader("Ask AI Analyst (Groq)")

            if st.button("Generate Explanation"):
                if not groq_api_key:
                    st.warning("Please enter your Groq API key in the sidebar.")
                else:
                    client = Groq(api_key=groq_api_key)

                    prompt = f"""
You are a cybersecurity analyst.

Prediction: {prediction}

Packet feature values:
{packet.to_string()}

Explain briefly and simply for a student:
1. Why this packet looks {prediction}
2. What these values indicate in network traffic
"""

                    with st.spinner("Groq AI is analyzing..."):
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.5
                        )

                        st.info(response.choices[0].message.content)

else:
    st.info("Click **Train Model Now** to start analysis.")


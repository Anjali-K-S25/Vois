import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**  
This system uses **Random Forest** to detect network attacks  
and **Groq AI** to explain packet behavior.
""")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (starts with gsk_)",
    type="password"
)
st.sidebar.caption("https://console.groq.com/keys")

st.sidebar.header("2. Model Training")

# --------------------------------------------------
# DATA LOADING (CLOUD-SAFE)
# --------------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(
            filepath,
            nrows=15000,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip",   # <-- SAFE replacement
            low_memory=False
        )

        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df["Label"] = df["Label"].astype(str)

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
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow Packets/s"
    ]

    target = "Label"

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None, 0, None, None, None

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return clf, acc, features, X_test, y_test

# --------------------------------------------------
# APP LOGIC
# --------------------------------------------------
df = load_data(DATA_FILE)

if df is None:
    st.stop()

st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

with st.expander("Preview Dataset"):
    st.dataframe(df.head())

if st.sidebar.button("Train Model Now"):
    with st.spinner("Training model..."):
        clf, acc, features, X_test, y_test = train_model(df)

        if clf:
            st.session_state.model = clf
            st.session_state.features = features
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.sidebar.success(f"Training Complete")
            st.sidebar.metric("Accuracy", f"{acc:.2%}")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
st.header("3. Threat Analysis Dashboard")

if "model" not in st.session_state:
    st.info("Click **Train Model Now** to begin.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traffic Simulation")
    st.info("Random packet is selected from test data.")

    if st.button("🎲 Capture Random Packet"):
        idx = np.random.randint(0, len(st.session_state.X_test))
        st.session_state.packet = st.session_state.X_test.iloc[idx]
        st.session_state.true_label = st.session_state.y_test.iloc[idx]

if "packet" in st.session_state:
    packet = st.session_state.packet

    with col1:
        st.subheader("Packet Features")
        st.dataframe(packet.to_frame(), use_container_width=True)

    with col2:
        st.subheader("Detection Result")

        prediction = st.session_state.model.predict(
            packet.to_frame().T
        )[0]

        if prediction == "BENIGN":
            st.success("STATUS: SAFE (BENIGN)")
        else:
            st.error(f"STATUS: ATTACK DETECTED ({prediction})")

        st.caption(f"Ground Truth Label: {st.session_state.true_label}")

        st.markdown("---")
        st.subheader("Ask AI Analyst (Groq)")

        if st.button("Generate Explanation"):
            if not groq_api_key:
                st.warning("Please enter Groq API key in sidebar.")
            else:
                try:
                    client = Groq(api_key=groq_api_key)

                    prompt = f"""
You are a cybersecurity analyst.

Prediction: {prediction}

Packet Features:
{packet.to_string()}

Explain briefly:
1. Why these values indicate the prediction
2. Whether traffic looks normal or malicious
3. Simple explanation for a student
"""

                    with st.spinner("Groq AI analyzing..."):
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.6
                        )

                        st.info(completion.choices[0].message.content)

                except Exception as e:
                    st.error(f"Groq API Error: {e}")

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
with st.expander("Model Evaluation Report"):
    preds = st.session_state.model.predict(st.session_state.X_test)
    report = classification_report(
        st.session_state.y_test, preds, output_dict=True
    )
    st.json(report)

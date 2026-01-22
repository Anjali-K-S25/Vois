import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from groq import Groq
import csv

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI-Based Network Intrusion Detection System", layout="wide")
st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**  
This system uses **Random Forest** to detect network attacks  
and **Groq AI** to explain packet behavior.
""")

DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.zip"

# -------------------------
# SIDEBAR
# -------------------------
groq_api_key = st.sidebar.text_input("Groq API Key (starts with gsk_)", type="password")
st.sidebar.caption("https://console.groq.com/keys")
st.sidebar.header("Model Training")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data(filepath):
    try:
        # Robust CSV loader
        df = pd.read_csv(filepath, encoding="latin1", low_memory=False)
        # Replace infinities and drop NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        # Auto-detect label column
        label_candidates = [c for c in df.columns if "label" in c.lower()]
        if not label_candidates:
            st.error("No label column found!")
            return None
        df[label_candidates[0]] = df[label_candidates[0]].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"File '{filepath}' not found.")
        return None
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return None

df = load_data(DATA_FILE)
if df is None:
    st.stop()
st.sidebar.success(f"Dataset Loaded: {len(df)} rows")
with st.expander("Preview Dataset"):
    st.dataframe(df.head())

# -------------------------
# TRAIN MODEL
# -------------------------
def train_model(df):
    # Detect label column
    label = [c for c in df.columns if "label" in c.lower()][0]
    # Select numeric features
    features = df.select_dtypes(include=np.number).columns.tolist()
    if label in features:
        features.remove(label)
    X = df[features]
    y = df[label]
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc, features, X_test, y_test

if st.sidebar.button("Train Model Now"):
    with st.spinner("Training model..."):
        clf, acc, features, X_test, y_test = train_model(df)
        st.session_state.model = clf
        st.session_state.features = features
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.sidebar.success("Training Complete")
        st.sidebar.metric("Accuracy", f"{acc:.2%}")

# -------------------------
# DASHBOARD
# -------------------------
st.header("Threat Analysis Dashboard")
if "model" not in st.session_state:
    st.info("Click **Train Model Now** to begin.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Traffic Simulation")
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
        prediction = st.session_state.model.predict(packet.to_frame().T)[0]
        if prediction.upper() == "BENIGN":
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
Prediction: {prediction}
Packet Features:
{packet.to_string()}
Explain briefly why this packet is classified this way for a student.
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

# -------------------------
# MODEL EVALUATION
# -------------------------
with st.expander("Model Evaluation Report"):
    preds = st.session_state.model.predict(st.session_state.X_test)
    report = classification_report(st.session_state.y_test, preds, output_dict=True)
    st.json(report)

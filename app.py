import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq

# --- PAGE SETUP ---
st.set_page_config(page_title="AI-NIDS Student Project", layout="wide")

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**  
Random Forest for attack detection + Groq AI for packet explanation.
""")

# --- CONFIG ---
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# --- SIDEBAR ---
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

st.sidebar.header("2. Model Training")

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath, nrows=15000)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def train_model(df):
    # SAFE feature list (exists in CIC-IDS)
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
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=42
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc, features, X_test, y_test

# --- LOAD DATA ---
df = load_data(DATA_FILE)
st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

if st.sidebar.button("Train Model Now"):
    clf, acc, features, X_test, y_test = train_model(df)
    st.session_state.update({
        "model": clf,
        "features": features,
        "X_test": X_test,
        "y_test": y_test
    })
    st.sidebar.success(f"Accuracy: {acc:.2%}")

# --- DASHBOARD ---
st.header("3. Threat Analysis Dashboard")

if "model" in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎲 Capture Random Packet"):
            idx = np.random.randint(len(st.session_state["X_test"]))
            st.session_state["packet"] = st.session_state["X_test"].iloc[idx]
            st.session_state["true_label"] = st.session_state["y_test"].iloc[idx]

    if "packet" in st.session_state:
        packet = st.session_state["packet"]

        with col1:
            st.subheader("Packet Features")
            st.dataframe(packet)

        with col2:
            prediction = st.session_state["model"].predict(
                packet.values.reshape(1, -1)
            )[0]

            if prediction == "BENIGN":
                st.success("SAFE (BENIGN)")
            else:
                st.error(f"ATTACK DETECTED: {prediction}")

            st.caption(f"Ground Truth: {st.session_state['true_label']}")

            if st.button("Generate AI Explanation"):
                client = Groq(api_key=groq_api_key)

                prompt = f"""
You are a cybersecurity analyst.
Prediction: {prediction}

Packet data:
{packet.to_string()}

Explain in simple student-friendly language.
"""

                res = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                )

                st.info(res.choices[0].message.content)
else:
    st.info("Train the model to start analysis.")

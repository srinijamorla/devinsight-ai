import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

from src.preprocess import clean_text
from src.model import load_model
from src.github_fetch import fetch_commits

# --- Hardcoded login credentials ---
USERNAME = "srinija_morla"
PASSWORD = "MSrinija@123"

# --- Page Config ---
st.set_page_config(page_title="DevInsight AI", layout="wide")

# --- Session State Auth Check ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- Login Page ---
if not st.session_state.authenticated:
    st.title("🔐 DevInsight AI – Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.stop()
        else:
            st.error("Invalid credentials")
    st.stop()

# --- Main App ---
st.title("🚀 DevInsight AI – Developer Commit Intelligence Dashboard")
model, vectorizer = load_model()

# --- Sidebar ---
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an Option", ["Upload File", "Analyze GitHub Repo"])

# --- Logout ---
if st.sidebar.button("🔓 Logout"):
    st.session_state.authenticated = False
    st.stop()

df = None

# --- Upload Excel ---
if option == "Upload File":
    st.subheader("📤 Upload a Commit File")
    uploaded_file = st.file_uploader("Upload an Excel file with a 'message' column", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if 'message' not in df.columns:
            st.error("Missing 'message' column.")
            df = None
        else:
            st.success("File uploaded successfully.")

# --- GitHub Repo Analysis ---
elif option == "Analyze GitHub Repo":
    st.subheader("🔗 Analyze Commits from GitHub")
    repo_url = st.text_input("Enter GitHub Repo URL (e.g., https://github.com/user/repo)")
    if st.button("Fetch Commits"):
        if repo_url:
            with st.spinner("Fetching and analyzing commits..."):
                try:
                    df = fetch_commits(repo_url)
                    if df.empty:
                        st.warning("No commits found or invalid repo.")
                        df = None
                    else:
                        st.success(f"Fetched {len(df)} commits.")
                except Exception as e:
                    st.error(f"Error fetching commits: {e}")
        else:
            st.warning("Please enter a valid URL.")

# --- Prediction and Visualization ---
if df is not None:
    df['cleaned'] = df['message'].apply(clean_text)
    X = vectorizer.transform(df['cleaned'])
    df['label'] = model.predict(X)

    if hasattr(model, "predict_proba"):
        df['confidence'] = model.predict_proba(X).max(axis=1)
    else:
        df['confidence'] = 0.0

    # --- Filters ---
    st.markdown("### 🎛️ Filter Predictions")
    unique_labels = sorted(df['label'].unique().tolist())
    selected_labels = st.multiselect("Filter by Label", unique_labels, default=unique_labels)
    confidence_range = st.slider("Filter by Confidence Score", 0.0, 1.0, (0.0, 1.0), step=0.01)

    filtered_df = df[
        (df['label'].isin(selected_labels)) &
        (df['confidence'] >= confidence_range[0]) &
        (df['confidence'] <= confidence_range[1])
    ]

    st.markdown("---")
    st.subheader("✅ Filtered Predictions")
    st.dataframe(filtered_df[['message', 'label', 'confidence']].head(10), use_container_width=True)

    st.subheader("📊 Filtered Prediction Summary")
    label_counts = filtered_df['label'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Filtered Commit Label Distribution")
    st.pyplot(fig)

    # --- Download ---
    output = BytesIO()
    filtered_df.to_excel(output, index=False)
    st.download_button(
        label="⬇️ Download Filtered Predictions",
        data=output.getvalue(),
        file_name="filtered_predicted_commits.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

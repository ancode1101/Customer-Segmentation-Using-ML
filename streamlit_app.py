# streamlit_app.py
# Run this file with:  streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIG & CONSTANTS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Define column constants for easy maintenance
TARGET_COL = 'PurchaseStatus'
NUM_COLS = [
    'Age', 'AnnualIncome', 'NumberOfPurchases',
    'TimeSpentOnWebsite', 'LoyaltyProgram', 'DiscountsAvailed'
]
CAT_COLS = ['Gender']
# Features for prediction model
FEATURES = NUM_COLS + CAT_COLS

st.set_page_config(page_title="Customer Insights Dashboard", layout="wide")
st.title("📊 Customer Insights & Purchase Prediction Dashboard")
st.markdown("**Built for Major Project 1 — with MySQL + ML Integration**")

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
# Use session state to store persistent data
if "df" not in st.session_state:
    st.session_state.df = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "scalers" not in st.session_state:
    st.session_state.scalers = {}
# ADD THIS: Track the name of the loaded data source
if "data_source_name" not in st.session_state:
    st.session_state.data_source_name = None

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (with Caching)
# -----------------------------------------------------------------------------

@st.cache_data
def load_from_mysql(host, user, password, database, table):
    """Connects to MySQL and fetches data."""
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    query = f"SELECT * FROM {table};"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data
def load_from_csv(file_path_or_buffer):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path_or_buffer)

@st.cache_data
def preprocess_data(_df):
    """Cleans and preprocesses the raw dataframe."""
    df = _df.copy()
    
    # Encode categorical columns
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'].astype(str))

    # Clean and impute numeric columns
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Clean target column
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)
    
    return df

@st.cache_data
def get_cluster_data(_df):
    """Prepares data for clustering (Scaling + PCA)."""
    try:
        cluster_features = [c for c in NUM_COLS if c in _df.columns]
        
        # 1. Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(_df[cluster_features])
        
        # 2. Apply PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        
        return X_scaled, coords, cluster_features, None
    except Exception as e:
        return None, None, None, f"Clustering pre-computation failed: {e}"

def eval_model(name, model, X_t, y_t, scaler=None):
    """Evaluates a single model and returns a metrics dictionary."""
    X_in = scaler.transform(X_t) if scaler else X_t
    y_pred = model.predict(X_in)
    y_prob = model.predict_proba(X_in)[:, 1]
    
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_t, y_pred),
        "Precision": precision_score(y_t, y_pred),
        "Recall": recall_score(y_t, y_pred),
        "F1": f1_score(y_t, y_pred),
        "AUC": roc_auc_score(y_t, y_prob)
    }

def run_model_training(_df):
    """Trains, evaluates, and saves prediction models."""
    try:
        X = _df[FEATURES]
        y = _df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create and fit scaler (only for LR)
        scaler_clf = StandardScaler().fit(X_train)
        X_train_s = scaler_clf.transform(X_train)

        # Initialize models
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)

        # Fit models
        lr.fit(X_train_s, y_train) # LR on scaled data
        rf.fit(X_train, y_train)   # RF on unscaled data

        # Evaluate
        results = pd.DataFrame([
            eval_model("Logistic Regression", lr, X_test, y_test, scaler=scaler_clf),
            eval_model("Random Forest", rf, X_test, y_test, scaler=None)
        ])

        # Save models and scaler to disk
        joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_model.joblib"))
        joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.joblib"))
        joblib.dump(scaler_clf, os.path.join(MODELS_DIR, "clf_scaler.joblib"))
        
        # Store in session state as well
        st.session_state.models['logistic_regression'] = lr
        st.session_state.models['random_forest'] = rf
        st.session_state.scalers['prediction_scaler'] = scaler_clf

        return results, "✅ Models trained and saved successfully."

    except Exception as e:
        return None, f"❌ Model training failed: {e}"

# -----------------------------------------------------------------------------
# SIDEBAR: Data Source
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Data Source Settings")
source = st.sidebar.radio("Select data source:", ["MySQL Database", "CSV File"], index=0)
status_placeholder = st.sidebar.empty()

try:
    if source == "MySQL Database":
        st.sidebar.subheader("MySQL Connection Details")
        host = st.sidebar.text_input("Host", "localhost")
        user = st.sidebar.text_input("User", "root")
        password = st.sidebar.text_input("Password", type="password")
        database = st.sidebar.text_input("Database", "RetailAnalytics")
        table = st.sidebar.text_input("Table", "Customers")

        if st.sidebar.button("Load data from MySQL"):
            with st.spinner("Connecting to MySQL..."):
                try:
                    raw_df = load_from_mysql(host, user, password, database, table)
                    st.session_state.df = preprocess_data(raw_df)
                    # Track the source
                    st.session_state.data_source_name = f"{database}.{table}"
                    status_placeholder.success(f"✅ Loaded {len(st.session_state.df)} rows from {database}.{table}")
                    st.rerun() # Force app to rerun with new data
                except Exception as e:
                    st.session_state.df = None
                    st.session_state.data_source_name = None
                    status_placeholder.error(f"❌ MySQL connection failed: {e}")
    
    else: # source == "CSV File"
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        
        # --- THIS IS THE KEY FIX ---
        # Only process if a file is uploaded AND it's not the one already in memory
        if uploaded and uploaded.name != st.session_state.data_source_name:
            with st.spinner(f"Loading {uploaded.name}..."):
                try:
                    raw_df = load_from_csv(uploaded)
                    st.session_state.df = preprocess_data(raw_df)
                    # Track the new file name
                    st.session_state.data_source_name = uploaded.name 
                    status_placeholder.success(f"✅ Loaded {len(st.session_state.df)} rows from {uploaded.name}")
                    st.rerun() # Rerun to apply new data
                except Exception as e:
                    st.session_state.df = None
                    st.session_state.data_source_name = None
                    status_placeholder.error(f"❌ Failed to read file: {e}")

        # Logic to load default CSV (only if nothing is loaded)
        elif st.session_state.df is None and not uploaded:
            default_csv = os.path.join(DATA_DIR, "customer_purchase_data.csv")
            if os.path.exists(default_csv):
                with st.spinner("Loading default CSV..."):
                    raw_df = load_from_csv(default_csv)
                    st.session_state.df = preprocess_data(raw_df)
                    st.session_state.data_source_name = "customer_purchase_data.csv"
            
    # --- ADDED STATUS PERSISTENCE ---
    # Show status based on what's in session state (for persistence)
    if st.session_state.data_source_name:
        if st.session_state.data_source_name == "customer_purchase_data.csv":
            status_placeholder.info("📂 Using default customer_purchase_data.csv.")
        else:
            status_placeholder.success(f"✅ Loaded {len(st.session_state.df)} rows from {st.session_state.data_source_name}")
    elif st.session_state.df is None:
        status_placeholder.warning("⚠️ No data loaded yet. Check sidebar.")


except Exception as e:
    st.error(f"Error while loading data: {e}")
    st.session_state.df = None
    st.session_state.data_source_name = None

# -----------------------------------------------------------------------------
# MAIN PAGE CONTENT
# -----------------------------------------------------------------------------

# Main check: Stop if no data is loaded
if st.session_state.df is None or st.session_state.df.empty:
    st.warning("⚠️ No data loaded yet. Please connect to MySQL or upload a CSV via the sidebar.")
    st.stop()

# Get persistent dataframe from session state
df = st.session_state.df

# Organize content into tabs
tab_preview, tab_viz, tab_cluster, tab_predict = st.tabs([
    "📄 Data Preview", "📊 Visualizations", "🎯 Clustering", "🤖 Prediction"
])

# -----------------------------------------------------------------------------
# TAB 1: DATA PREVIEW
# -----------------------------------------------------------------------------
with tab_preview:
    st.subheader("Data Preview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())
    
    st.subheader("Data Summary Statistics")
    st.dataframe(df[NUM_COLS].describe())

# -----------------------------------------------------------------------------
# TAB 2: VISUALIZATIONS
# -----------------------------------------------------------------------------
with tab_viz:
    st.subheader("Customer Demographics")
    col1, col2 = st.columns(2)
    with col1:
        if 'AnnualIncome' in df.columns:
            fig = px.histogram(df, x='AnnualIncome', nbins=30, title="Annual Income Distribution")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'Gender' in df.columns:
            # Note: 'Gender' is now 0/1. We can map it back for the chart.
            gender_map = {0: "Female", 1: "Male"}
            df_viz = df.copy()
            df_viz['Gender_Label'] = df_viz['Gender'].map(gender_map).fillna("Unknown")
            fig2 = px.pie(df_viz, names='Gender_Label', title="Gender Distribution")
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Purchase Behavior Analysis")
    if 'AnnualIncome' in df.columns and 'TimeSpentOnWebsite' in df.columns:
        color_col = TARGET_COL if TARGET_COL in df.columns else None
        fig3 = px.scatter(df, x='AnnualIncome', y='TimeSpentOnWebsite',
                          color=df[color_col].astype(str) if color_col else None,
                          title="Income vs Time Spent on Website (Colored by Purchase Status)")
        st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: K-MEANS CLUSTERING
# -----------------------------------------------------------------------------
with tab_cluster:
    st.subheader("Customer Segmentation with K-Means")
    
    # 1. Get cached scaled data and PCA coordinates
    X_scaled, coords, cluster_features, error = get_cluster_data(df)
    
    if error:
        st.error(error)
    else:
        # 2. Add interactive slider
        k_val = st.slider("Select number of clusters (k)", 2, 10, 4)
        
        # 3. Fit K-Means (This is fast and runs on slider change)
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 4. Create plot dataframe
        plot_df = pd.DataFrame(coords, columns=['pca1', 'pca2'])
        plot_df['cluster'] = cluster_labels.astype(str)
        for i, col in enumerate(cluster_features):
             plot_df[col] = df[col]
             
        # 5. Plot
        fig4 = px.scatter(plot_df, x='pca1', y='pca2',
                          color='cluster',
                          hover_data=cluster_features,
                          title="Customer Clusters (2D PCA View)")
        st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: SUPERVISED LEARNING & PREDICTION
# -----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Purchase Prediction Models")
    
    if TARGET_COL not in df.columns:
        st.warning(f"⚠️ '{TARGET_COL}' column not found — skipping model training.")
    else:
        # On-demand model training
        if st.button("Train Purchase Prediction Models"):
            with st.spinner("Training models... This may take a moment."):
                results_df, message = run_model_training(df)
            
            if results_df is not None:
                st.success(message)
                st.dataframe(results_df)
            else:
                st.error(message)

    st.markdown("---")
    st.subheader("🧮 Predict New Customer Purchase")

    model_path_lr = os.path.join(MODELS_DIR, "logistic_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "clf_scaler.joblib")

    # Check if models exist on disk before showing the form
    if not (os.path.exists(model_path_lr) and os.path.exists(scaler_path)):
        st.info("Models not found. Please click the 'Train Models' button above to train and save them first.")
    else:
        try:
            with st.form("prediction_form"):
                st.markdown("**Enter customer details:**")
                
                # Use columns for a cleaner layout
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input("Age", 18, 80, 30)
                    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                    income = st.number_input("Annual Income ($)", 10000, 200000, 50000, step=1000)
                with c2:
                    purchases = st.number_input("Number of Purchases", 0, 50, 5)
                    time = st.number_input("Time Spent on Website (mins)", 0.0, 100.0, 10.0, step=0.5)
                    loyalty = st.selectbox("In Loyalty Program?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                with c3:
                    discounts = st.number_input("Discounts Availed", 0, 20, 2)
                
                submitted = st.form_submit_button("Predict Purchase")

            if submitted:
                # Load models and scaler from disk
                lr = joblib.load(model_path_lr)
                rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
                scaler_clf = joblib.load(scaler_path)

                # Create input dataframe
                new_df = pd.DataFrame([[age, gender, income, purchases, time, loyalty, discounts]], columns=FEATURES)
                
                # Scale for LR
                new_scaled = scaler_clf.transform(new_df)
                
                # Predict
                pred_lr = lr.predict(new_scaled)[0]
                prob_lr = lr.predict_proba(new_scaled)[0][1]
                pred_rf = rf.predict(new_df)[0]
                prob_rf = rf.predict_proba(new_df)[0][1]

                st.subheader("Prediction Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Logistic Regression Prediction",
                              "Will Purchase" if pred_lr == 1 else "Won't Purchase",
                              f"{prob_lr*100:.1f}% Probability")
                with res_col2:
                    st.metric("Random Forest Prediction",
                              "Will Purchase" if pred_rf == 1 else "Won't Purchase",
                              f"{prob_rf*100:.1f}% Probability")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Major Project 1 © 2025")
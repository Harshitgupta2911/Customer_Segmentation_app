import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import hashlib

# --- INITIALIZE SESSION STATE ---
# We use st.session_state to persist data (like the ML model) across app reruns.
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'Y_test' not in st.session_state:
    st.session_state.Y_test = None
if 'data_hash' not in st.session_state:
    st.session_state.data_hash = None

# --- MODEL EVALUATION ---


def evaluate_model(model, X_test, Y_test):
    """Calculates and displays model accuracy and classification metrics."""
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    st.subheader("📊 Model Performance")
    st.write(f"Accuracy: {acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(Y_test, Y_pred))

# --- MODEL TRAINING ---


def build_model(rfm_df):
    """Trains a Random Forest Classifier to predict if a customer will Churn (1) or not (0)."""
    df = rfm_df.copy()
    # Target variable: Churn is 1 if Recency > 90 days, else 0
    df["Churn"] = df["Recency"].apply(lambda x: 1 if x > 90 else 0)

    # Feature selection for training
    X = df[["Frequency", "Monetary", "Average Order Revenue",
            "Frequency per Month", "Average Time Between Orders"]]
    Y = df["Churn"]

    # Split into 80% training and 20% testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # Initialize and fit the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    return model, X_test, Y_test

# --- PREDICTION AND VISUALIZATION FOR INDIVIDUAL CUSTOMERS ---


def churn_prediction(rfm_df, model, cust_id=None):
    """Predicts churn risk for a specific ID and highlights them on a scatter plot."""
    col1, col2 = st.columns(2)
    temp_df = rfm_df.copy()

    if cust_id:
        customer = rfm_df[rfm_df["CustomerID"] == int(cust_id)]
        if not customer.empty:
            # Prepare inputs for prediction
            frequency = customer["Frequency"].values[0]
            monetary = customer["Monetary"].values[0]
            aor = customer["Average Order Revenue"].values[0]
            fpm = customer["Frequency per Month"].values[0]
            ato = customer["Average Time Between Orders"].values[0]

            input_data = pd.DataFrame([[frequency, monetary, aor, fpm, ato]],
                                      columns=["Frequency", "Monetary", "Average Order Revenue",
                                               "Frequency per Month", "Average Time Between Orders"])

            # Predict probability
            prob = model.predict_proba(input_data)

            with col1:
                st.subheader("Customer Details")
                st.write(customer)
                # Check if 'Churn' class exists in model training data
                if 1 in model.classes_:
                    churn_index = list(model.classes_).index(1)
                    churn_prob = prob[0][churn_index]
                else:
                    churn_prob = 0

                # UI feedback based on probability threshold (50%)
                if churn_prob > 0.5:
                    st.error(f"⚠️ High Risk of Churn ({churn_prob*100:.2f}%)")
                    st.warning(
                        "👉 Suggestion: Offer discount or re-engagement campaign")
                else:
                    st.success(f"✅ Low Risk ({churn_prob*100:.2f}%)")

            with col2:
                # Visualize the specific customer among the population
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(
                    data=temp_df, x="Frequency", y="Monetary",
                    hue="ChurnLabel", palette={"Yes": "red", "No": "green"},
                    alpha=0.5, ax=ax
                )
                # Add the 'Yellow Star' highlight for the searched customer
                ax.scatter(frequency, monetary, color="yellow", s=200,
                           edgecolor="black", label="Selected Customer")
                ax.legend()
                st.pyplot(fig)
        else:
            st.error("Customer not found")
    else:
        st.warning("Please enter a Customer ID")

# --- GLOBAL CHURN VISUALIZATION ---


def churn(rfm_df):
    """Generates a bar chart of the Churn count."""
    fig, ax = plt.subplots()
    sns.countplot(data=rfm_df, x="ChurnLabel", ax=ax)
    st.pyplot(fig)

# --- CLUSTERING (K-MEANS) ---


@st.cache_data
def clustering(rfm_df):
    """Uses K-Means to group customers into 4 distinct clusters."""
    rfm_clustering = rfm_df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clustering)

    # Optional Elbow Method inertia calculation (not used for logic but kept for consistency)
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        inertia.append(kmeans.inertia_)

    # Fit final model with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm_df

# --- PARETO (80/20) ANALYSIS ---


def top_customers(rfm_df):
    """Finds what % of customers contribute to 80% of total revenue."""
    top_df = rfm_df.sort_values(by='Monetary', ascending=False)
    total_revenue = rfm_df['Monetary'].sum()
    top_df['cum_revenue'] = top_df['Monetary'].cumsum()
    top_df["cum_revenue_pct"] = (top_df['cum_revenue']/total_revenue)*100
    top_customers = top_df[top_df['cum_revenue_pct'] <= 80]
    return len(top_customers)/len(rfm_df)*100

# --- SEGMENTATION RULESET ---


def segment_customer(row):
    """Assigns labels based on RFM scores (1-5)."""
    if row['R_score'] >= 4 and row['F_score'] >= 4 and row['M_score'] >= 4:
        return "Champions"
    elif row['F_score'] >= 4:
        return "Loyal Customers"
    elif row['R_score'] >= 4:
        return "Recent Customers"
    elif row['R_score'] <= 2:
        return "At Risk"
    else:
        return "Others"

# --- CORE RFM LOGIC ---


@st.cache_data
def Analysis(data):
    """The engine that calculates Recency, Frequency, and Monetary metrics."""
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
    data = data.dropna(subset=['InvoiceDate'])
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
    reference_data = data["InvoiceDate"].max()

    # Recency Calculation
    last_purchase_date = data.groupby("CustomerID")["InvoiceDate"].max()
    recency = reference_data - last_purchase_date
    recency_df = recency.reset_index()
    recency_df.columns = ["CustomerID", "Recency"]
    recency_df['Recency'] = recency_df['Recency'].dt.days

    # Revenue (Monetary) and Order Count (Frequency)
    revenue = data.groupby("CustomerID")["TotalPrice"].sum()
    orders = data.groupby("CustomerID")["InvoiceNo"].nunique()

    # Average Order Revenue (AOR)
    AOR = revenue/orders
    AOR_df = AOR.reset_index()
    AOR_df.columns = ["CustomerID", "Average Order Revenue"]

    # Frequency Per Month (FPM)
    first_purchase_date = data.groupby("CustomerID")["InvoiceDate"].min()
    total_days = (reference_data - first_purchase_date).dt.days
    months = (total_days / 30).apply(lambda x: x if x > 0 else 1/30)
    FPM = orders/months
    FPM_df = FPM.reset_index()
    FPM_df.columns = ["CustomerID", "Frequency per Month"]

    # Time Between Orders
    data = data.sort_values(by=['CustomerID', 'InvoiceNo'])
    orders_df = data[['CustomerID', 'InvoiceNo',
                      'InvoiceDate']].drop_duplicates()
    orders_df['prev_date'] = orders_df.groupby(
        'CustomerID')['InvoiceDate'].shift(1)
    orders_df['days_diff'] = (
        orders_df['InvoiceDate'] - orders_df['prev_date']).dt.days
    avg_time = orders_df.groupby('CustomerID')['days_diff'].mean()
    avg_time_df = avg_time.reset_index().rename(
        columns={'days_diff': 'Average Time Between Orders'})

    # Merge all metrics into Master RFM dataframe
    frequency_df = orders.reset_index().rename(
        columns={'InvoiceNo': 'Frequency'})
    monetary_df = revenue.reset_index().rename(
        columns={'TotalPrice': 'Monetary'})

    rfm_df = pd.merge(recency_df, frequency_df, on='CustomerID')
    rfm_df = pd.merge(rfm_df, monetary_df, on='CustomerID')
    rfm_df = pd.merge(rfm_df, FPM_df, on='CustomerID')
    rfm_df = pd.merge(rfm_df, AOR_df, on='CustomerID')
    rfm_df = pd.merge(rfm_df, avg_time_df, on='CustomerID')

    # Assign Scores (Quintiles)
    r_bins = pd.qcut(rfm_df['Recency'], 5, duplicates='drop')
    rfm_df['R_score'] = 5 - r_bins.cat.codes
    f_bins = pd.qcut(rfm_df['Frequency'], 5, duplicates='drop')
    rfm_df['F_score'] = f_bins.cat.codes + 1
    m_bins = pd.qcut(rfm_df['Monetary'], 5, duplicates='drop')
    rfm_df['M_score'] = m_bins.cat.codes + 1

    rfm_df['Average Time Between Orders'] = pd.to_numeric(
        rfm_df['Average Time Between Orders'], errors='coerce').fillna(0)

    # Calculate Segment and Churn Labels
    rfm_df['Segment'] = rfm_df.apply(segment_customer, axis=1)
    rfm_df["Churn"] = rfm_df["Recency"].apply(lambda x: 1 if x > 90 else 0)
    rfm_df['ChurnLabel'] = rfm_df['Churn'].map({0: 'No', 1: 'Yes'})
    return rfm_df

# --- DATA CLEANING ---


@st.cache_data
def preprocessing(data):
    """Cleans data by removing negatives, nulls, and duplicates."""
    data = data[data['Quantity'] >= 0]
    data = data.dropna(subset=['Description', 'CustomerID'])
    data['CustomerID'] = data['CustomerID'].astype(int)
    if data.duplicated().sum() > 0:
        data.drop_duplicates(inplace=True)
    return data

# --- FILE LOADING ---


@st.cache_data
def load_data(file, file_type):
    """Handles CSV or Excel ingestion."""
    if file_type == "CSV":
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


# --- STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="Customer Dashboard", layout="wide")
st.title("📊 Customer Data Analysis Dashboard")

# Sidebar
st.sidebar.header("Upload Data")
file_type = st.sidebar.selectbox("Select File Type", ["CSV", "Excel"])

if file_type == "CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
else:
    uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

# Logic Flow
if uploaded_file is not None:
    with st.spinner("⏳ Loading your data..."):
        data = load_data(uploaded_file, file_type)

    st.success("✅ Data Loaded Successfully!")
    st.subheader("📄 Data Preview")
    st.dataframe(data.head())

    # Metadata summary
    st.subheader("📊 Data Summary")
    st.write(
        f"**Number of Rows:** {data.shape[0]} | **Number of Columns:** {data.shape[1]}")
    st.write("**Column Names:**", data.columns.tolist())

    # Process and Analyze
    st.header("🔍 Data Analysis Results")
    preprocesses_data = preprocessing(data)
    rfm_df = Analysis(preprocesses_data)
    rfm_df = clustering(rfm_df)
    st.dataframe(rfm_df)

    # Segmentation Charts
    st.header("Customer Segmentation Visualization")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar(rfm_df['Segment'].value_counts().index,
               rfm_df['Segment'].value_counts().values)
        ax.set_title('Customer Segments')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(rfm_df['Segment'].value_counts(
        ).values, labels=rfm_df['Segment'].value_counts().index, autopct='%1.1f%%')
        ax.set_title('Customer Segments Distribution')
        st.pyplot(fig)

    st.write(
        f"{top_customers(rfm_df):.2f}% Customers Contributing to 80% of Revenue")

    # Feature Relationship Selector
    st.header("Visualization of Feature Relationships")
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("Select X-axis Feature", ['Recency', 'Frequency', 'Monetary',
                                 'Average Order Revenue', 'Frequency per Month', 'Average Time Between Orders'])
        feature_y = st.selectbox("Select Y-axis Feature", ['Recency', 'Frequency', 'Monetary',
                                 'Average Order Revenue', 'Frequency per Month', 'Average Time Between Orders'])
        if st.button("Visualize"):
            with col2:
                fig, ax = plt.subplots()
                ax.scatter(rfm_df[feature_x], rfm_df[feature_y],
                           marker='.', linestyle='', alpha=0.7, color='red')
                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                st.pyplot(fig)

    # K-Means Visuals
    st.header("Customer Clustering Using K-Means")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(data=rfm_df, x="Recency", y="Monetary",
                        hue="Cluster", palette="Set1", ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=rfm_df, x="Frequency", y="Monetary",
                        hue="Cluster", palette="Set1", ax=ax)
        st.pyplot(fig)

    # Churn Visuals
    st.header("Churn Summary")
    col1, col2 = st.columns(2)
    with col1:
        churn(rfm_df)
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=rfm_df, x="Recency", y="Monetary", hue="ChurnLabel", palette={
                        "Yes": "red", "No": "green"}, ax=ax)
        st.pyplot(fig)

    # ML Hashing & Training Logic
    data_hash = hashlib.md5(rfm_df.to_csv(index=False).encode()).hexdigest()
    if st.session_state.data_hash != data_hash:
        with st.spinner("🤖 Training model..."):
            model, X_test, Y_test = build_model(rfm_df)
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.Y_test = Y_test
            st.session_state.data_hash = data_hash

    # Prediction Interface
    if st.session_state.model is not None:
        st.header("🎯 Individual Customer Churn Prediction")
        cust_id = st.number_input(
            "Customer ID", format="%d", min_value=0, step=1, key="churn_input")
        if st.checkbox("Show Model Performance"):
            evaluate_model(st.session_state.model,
                           st.session_state.X_test, st.session_state.Y_test)
        if st.button("Predict Churn"):
            with st.spinner("🔍 Analyzing..."):
                churn_prediction(rfm_df, st.session_state.model, int(cust_id))
    else:
        st.info("💡 The prediction model will be available once data is processed.")
else:
    st.info("👈 Please upload a file from the sidebar to begin.")

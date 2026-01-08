# --------- IMPORT LIBRARIES ---------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --------- PAGE CONFIG ---------
st.set_page_config(page_title="Customer Analytics Platform", layout="wide")
st.title('Customer Churn Prediction and Sales Dashboard')

# --------- SIDEBAR NAVIGATION ---------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📂 Data Explorer",
        "🤖 Churn Prediction",
        "🧩 Customer Segmentation",
        "📈 Sales Dashboard",
        "📊 Business Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload CSV Files**")
customer_file = st.sidebar.file_uploader("Customer Data", type=["csv"])
transaction_file = st.sidebar.file_uploader("Transaction Data", type=["csv"])

# --------- LOAD DATA ---------
def load_data(file):
    return pd.read_csv(file)

if customer_file and transaction_file:
    customers = load_data(customer_file)
    transactions = load_data(transaction_file)

    # ========== COMMON PREPROCESSING ==========
    customers['Churn'] = customers['Churn'].astype(int)

    cat_cols = customers.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        customers[col] = le.fit_transform(customers[col].astype(str))

    X = customers.drop(columns=['Churn'])
    y = customers['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # =============================
    # 🏠 OVERVIEW DASHBOARD
    # =============================
    if page == "🏠 Overview":
        st.subheader("🏠 Project Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", customers.shape[0])
        col2.metric("Churn Rate (%)", round(y.mean() * 100, 2))
        col3.metric("Total Revenue", round(transactions['amount'].sum(), 2))

        st.markdown("""
        ### About This Project
        - Predicts customer churn using ML & DL models
        - Analyzes sales trends over time
        - Segments customers for targeted marketing
        - Built with **Python + Streamlit**
        """)

    # =============================
    # 📂 DATA EXPLORER DASHBOARD
    # =============================
    elif page == "📂 Data Explorer":
        st.subheader("📂 Data Explorer")
        st.subheader("Customer Dataset")
        st.dataframe(customers.head(100))

        st.subheader("Transaction Dataset")
        st.dataframe(transactions.head(100))

    # =============================
    # 🤖 CHURN PREDICTION DASHBOARD
    # =============================
    elif page == "🤖 Churn Prediction":
        st.subheader("🤖 Churn Prediction Models")

        model_name = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "ANN"]
        )

        preds = None

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=6)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        else:
            ann = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            ann.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
            preds = (ann.predict(X_test) > 0.5).astype(int).ravel()

        # ----- MODEL METRICS -----
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", round(acc * 100, 2))
        col2.metric("Precision", round(report['1']['precision'] * 100, 2))
        col3.metric("Recall", round(report['1']['recall'] * 100, 2))
        col4.metric("F1 Score", round(report['1']['f1-score'] * 100, 2))

        st.subheader("📄 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

    elif page == "🤖 Churn Prediction":
        st.title("🤖 Churn Prediction Models")

        model_name = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "ANN"]
        )

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=6)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=200)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

        else:
            ann = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            ann.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
            _, acc = ann.evaluate(X_test, y_test, verbose=0)

        st.metric("Model Accuracy", round(acc * 100, 2))

    # =============================
    # 🧩 CUSTOMER SEGMENTATION
    # =============================
    elif page == "🧩 Customer Segmentation":
        st.subheader("🧩 Customer Segmentation")
        k = st.slider("Number of Clusters", 2, 8, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        customers['Cluster'] = kmeans.fit_predict(X_scaled)

        fig, ax = plt.subplots()
        sns.countplot(x='Cluster', data=customers, ax=ax)
        st.pyplot(fig)

    # =============================
    # 📈 SALES DASHBOARD
    # =============================
    elif page == "📈 Sales Dashboard":
        st.subheader("📈 Sales Trend Dashboard")
        transactions['date'] = pd.to_datetime(transactions['date'])
        transactions['month'] = transactions['date'].dt.to_period('M')
        monthly_sales = transactions.groupby('month')['amount'].sum().reset_index()
        monthly_sales['month'] = monthly_sales['month'].astype(str)

        fig, ax = plt.subplots()
        ax.plot(monthly_sales['month'], monthly_sales['amount'])
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        ax.set_title("Monthly Sales Trend")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # =============================
    # 📊 BUSINESS INSIGHTS
    # =============================
    elif page == "📊 Business Insights":
        st.subheader("📊 Business Insights & Churn Reasons")

        churned_df = customers[customers['Churn'] == 1]
        retained_df = customers[customers['Churn'] == 0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", customers.shape[0])
        col2.metric("Churned Customers", churned_df.shape[0])
        col3.metric("Churn Rate (%)", round(customers['Churn'].mean() * 100, 2))

        # =============================
        # 📉 WHY CUSTOMERS ARE LEAVING (VISUAL)
        # =============================
        st.subheader("📉 Why Customers Are Leaving")

        reason_cols = [col for col in customers.columns if col not in ['Churn', 'Cluster']]
        churn_reason_summary = churned_df[reason_cols].mean().sort_values(ascending=False)

        colA, colB = st.columns(2)

        # Bar Chart – Top Churn Drivers
        with colA:
            fig1, ax1 = plt.subplots()
            churn_reason_summary.head(6).plot(kind='barh', ax=ax1)
            ax1.set_title("Top Churn Drivers")
            ax1.set_xlabel("Relative Impact")
            st.pyplot(fig1)

        # Heatmap – Churn Correlation
        with colB:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            corr = customers.corr()
            sns.heatmap(corr[['Churn']].sort_values(by='Churn', ascending=False),
                        annot=True, cmap='coolwarm', ax=ax2)
            ax2.set_title("Correlation with Churn")
            st.pyplot(fig2)

        # =============================
        # 📊 CHURN DISTRIBUTION BY FEATURES
        # =============================
        st.subheader("📊 Churn Distribution by Key Features")

        colX, colY = st.columns(2)

        with colX:
            fig3, ax3 = plt.subplots()
            sns.boxplot(x='Churn', y=customers.columns[1], data=customers, ax=ax3)
            ax3.set_title("Feature vs Churn")
            st.pyplot(fig3)

        with colY:
            fig4, ax4 = plt.subplots()
            sns.histplot(data=customers, x=customers.columns[1], hue='Churn', kde=True, ax=ax4)
            ax4.set_title("Churned vs Retained Distribution")
            st.pyplot(fig4)

        st.markdown("""
        ### Key Business Reasons for Customer Churn
        - **High Monthly Charges** → Customers feel the service is expensive
        - **Low Tenure** → New customers are more likely to leave
        - **Short-Term Contracts** → Monthly contracts churn more
        - **Low Engagement / Purchase Frequency**
        - **Payment Friction** → Non-automated payment methods

        ### 🎯 Recommended Business Actions
        - Early intervention for new customers
        - Loyalty discounts for high-value users
        - Contract upgrades (Monthly → Yearly)
        - Personalized churn-prevention campaigns
        """)


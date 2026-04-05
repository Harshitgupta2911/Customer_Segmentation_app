# 📊 Customer RFM Analysis & Churn Prediction Dashboard

A professional, interactive Streamlit dashboard that transforms raw e-commerce transaction data into actionable business insights. This application uses **RFM (Recency, Frequency, Monetary) Analysis**, **K-Means Clustering**, and **Machine Learning (Random Forest)** to segment customers and predict churn risk.

## 🚀 Features

- **Data Preprocessing:** Automatically cleans data, handles missing values, and removes invalid transactions.
- **RFM Analysis:** Calculates core metrics including Recency, Frequency, and Monetary value for every customer.
- **Deep Insights:** Computes advanced metrics like *Average Order Revenue*, *Frequency per Month*, and *Average Time Between Orders*.
- **Customer Segmentation:** Categorizes customers into segments like *Champions*, *Loyal Customers*, and *At Risk*.
- **Automated Clustering:** Uses K-Means machine learning to find hidden patterns and group similar customers.
- **Churn Prediction:** Trains a Random Forest Classifier to predict the probability of a customer leaving.
- **Interactive Visualizations:** Dynamic bar charts, pie charts, and scatter plots powered by Seaborn and Matplotlib.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest, K-Means, StandardScaler)
- **Visualization:** Matplotlib, Seaborn

---

## 📋 Data Requirements

To use this dashboard, upload a **CSV** or **Excel** file containing the following columns:

| Column Name | Description |
| :--- | :--- |
| `CustomerID` | Unique identifier for each customer |
| `InvoiceNo` | Unique identifier for each transaction |
| `InvoiceDate` | Date of the transaction (Format: YYYY-MM-DD) |
| `Quantity` | Number of items purchased |
| `UnitPrice` | Price per single unit of the product |

---

## ⚙️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Harshitgupta2911/Customer_Segmentation_app.git](https://github.com/Harshitgupta2911/Customer_Segmentation_app.git)
   cd Customer_Segmentation_app

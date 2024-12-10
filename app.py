pip install streamlit pandas numpy scikit-learn matplotlib yfinance

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Sample model training (use your trading model instead)
@st.cache
def train_model():
    # Load some example data
    df = yf.download("AAPL", start="2015-01-01", end="2023-01-01")
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'] > 0).astype(int)
    df = df.dropna()

    X = df[['Close', 'Volume']].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy, df

model, accuracy, df = train_model()

# Streamlit App Interface
st.title("Stock Trading Prediction App")
st.write("This app predicts stock price movements based on historical data.")

# Sidebar for User Input
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch Data
@st.cache
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = fetch_data(ticker, start_date, end_date)

# Display Data
st.subheader(f"Stock Data for {ticker}")
st.dataframe(data)

# Add Features
if not data.empty:
    data['Return'] = data['Close'].pct_change()
    data['Signal'] = model.predict(data[['Close', 'Volume']].fillna(0).values)
    st.subheader("Predictions")
    st.dataframe(data[['Close', 'Return', 'Signal']])

    # Visualize
    st.subheader("Close Price with Predicted Signals")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Close Price')
    ax.scatter(data.index, data['Close'], c=data['Signal'], cmap='coolwarm', label='Signal', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# Show Model Accuracy
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"Accuracy: {accuracy * 100:.2f}%")

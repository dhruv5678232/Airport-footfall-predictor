import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from GitHub
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
df = pd.read_csv(file_url)

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Feature Engineering
# Convert 'date' to datetime for potential use in the model
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Define required columns
required_columns = ["airport", "season", "date", "is_weekend", "load_factor (%)"]

if all(col in df.columns for col in required_columns):
    # Convert categorical columns to numeric codes
    categorical_cols = ["airport", "season", "is_weekend"]
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # Convert 'date' to numeric representation (ordinal)
    df['date_numeric'] = df['date'].map(lambda x: x.toordinal())

    # Select Relevant Features
    features = ["airport", "season", "date_numeric", "is_weekend", "load_factor (%)"]
    target = "actual_footfall"

    # Train-test split (80-20)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display Model Performance
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")
    st.sidebar.write(f"R² Score: {r2:.2f}")
    st.sidebar.success("Model Trained Successfully ✅")

    # Visualization Column
    st.subheader("Data Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=df["date

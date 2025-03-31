import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset from GitHub
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
try:
    df = pd.read_csv(file_url)
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase

    # Check for missing columns
    required_columns = ["airport", "season", "flight type", "year", "weekday/weekend", "load_factor"]
    if all(col in df.columns for col in required_columns):
        # Feature Engineering
        categorical_cols = ["airport", "season", "flight type", "weekday/weekend"]
        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes  # Convert categories to numbers

        # Select Relevant Features
        features = ["airport", "season", "flight type", "year", "weekday/weekend", "load_factor"]
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
        sns.lineplot(x=df["year"], y=df["actual_footfall"], marker="o", label="Actual Footfall", ax=ax)
        sns.lineplot(x=df["year"], y=df["predicted_footfall"], marker="s", label="Predicted Footfall", ax=ax)
        plt.xlabel("Year")
        plt.ylabel("Footfall")
        plt.legend()
        st.pyplot(fig)

    else:
        missing_cols = [col for col in required_columns if col not in df.columns]
        st.error(f"Missing required columns: {', '.join(missing_cols)}")

except Exception as e:
    st.error(f"Error loading dataset: {e}")

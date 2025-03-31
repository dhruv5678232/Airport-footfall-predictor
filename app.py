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
    st.write("Dataset Loaded Successfully")
    st.write(df.head())  # Display first few rows for debugging
    
    # Ensure required columns exist
    required_columns = ["Airport", "Season", "Year"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
    else:
        # Extract unique values safely
        airports = df["Airport"].dropna().unique().tolist()
        seasons = df["Season"].dropna().unique().tolist() if "Season" in df else ["Summer", "Monsoon", "Winter"]
        flight_types = ["Domestic", "International"]
        years = sorted(df["Year"].dropna().unique().tolist()) if "Year" in df else []
        weekday_options = ["Weekday", "Weekend"]
        
        # Streamlit UI
        st.title("Airport Footfall Prediction")
        st.sidebar.header("Input Parameters")
        
        selected_airport = st.sidebar.selectbox("Select Airport:", airports if airports else ["No data available"])
        selected_season = st.sidebar.selectbox("Select Season:", seasons)
        selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)
        selected_year = st.sidebar.slider("Select Year:", min_value=min(years) if years else 2020, max_value=max(years) if years else 2030, step=1)
        selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)
        
        # Display selected inputs
        st.write("### Selected Inputs")
        st.write(f"*Airport:* {selected_airport}")
        st.write(f"*Season:* {selected_season}")
        st.write(f"*Flight Type:* {selected_flight_type}")
        st.write(f"*Year:* {selected_year}")
        st.write(f"*Day Type:* {selected_weekday}")
        
        # ✅ Select Relevant Features
        features = ["Airport", "Season", "Flight Type", "Year", "Weekday/Weekend", "Load_Factor"]
        target = "Actual_Footfall"

        # ✅ Ensure all required columns exist
        if all(col in df for col in features + [target]):
            # Train-test split (80-20)
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # ✅ Train Model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # ✅ Make Predictions
            y_pred = model.predict(X_test)

            # ✅ Evaluate Model
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # ✅ Display Model Performance
            st.sidebar.subheader("Model Performance")
            st.sidebar.write(f"MAE: {mae:.2f}")
            st.sidebar.write(f"RMSE: {rmse:.2f}")
            st.sidebar.write(f"R² Score: {r2:.2f}")

            st.sidebar.success("Model Trained Successfully ✅")
        else:
            st.sidebar.error("Missing columns required for model training.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")

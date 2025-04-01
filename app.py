import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_url = "https://github.com/dhruv5678232/Airport-footfall-predictor/raw/main/Airport_Flight_Data_Cleaned.csv"
try:
    df = pd.read_csv(file_url)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Data Preprocessing
df.columns = df.columns.str.lower()  # Ensure lowercase column names
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Parse dates
df = df.dropna(subset=['date'])  # Remove rows with invalid dates

# Rename columns for consistency
df.rename(columns={
    "load_factor_(%)": "load_factor",
    "is_weekend": "weekday_weekend"
}, inplace=True)

# Create 'flight_type' column (0 for Domestic, 1 for International)
df['flight_type'] = (df['domestic_flights'] == 0).astype(int)

# Extract unique values for UI inputs
airports = df["airport"].dropna().unique().tolist()
seasons = df["season"].dropna().unique().tolist()
flight_types = ["Domestic", "International"]
weekday_options = ["Weekday", "Weekend"]

# Debug the 'year' column
if "year" not in df.columns:
    st.error("The 'year' column is missing from the dataset.")
    st.stop()

# Ensure 'year' column is numeric and handle missing values
df["year"] = pd.to_numeric(df["year"], errors='coerce')
df = df.dropna(subset=["year"])  # Remove rows with invalid years
years = sorted(df["year"].unique().tolist())

# Debug output
st.sidebar.write("Debug: Unique years in dataset =", years)
st.sidebar.write("Debug: Number of unique years =", len(years))

weather_options = ["Bad", "Good"]
temperature_min, temperature_max = int(df["temperature"].min()), int(df["temperature"].max())
peak_season_options = ["No", "Yes"]
holiday_options = ["No", "Yes"]

# Compute mean values for removed features
mean_total_flights = int(df["total_flights"].mean())
mean_load_factor = df["load_factor"].mean()
mean_economic_trend = df["economic_trend"].mean()

# Streamlit UI
st.title("Airport Footfall Prediction and Analysis")
st.sidebar.header("Input Parameters for Prediction")

# User inputs for remaining factors
selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)

# Handle the year selection
if len(years) == 0:
    st.sidebar.error("No valid years found in the dataset.")
    st.stop()
elif len(years) == 1:
    selected_year = int(years[0])  # Ensure integer
    st.sidebar.write(f"Year: {selected_year} (Only one year available in the dataset)")
else:
    selected_year = st.sidebar.slider("Select Year:", min_value=int(min(years)), max_value=int(max(years)), step=1)

selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)
selected_temperature = st.sidebar.slider("Select Temperature (°C):", min_value=temperature_min, max_value=temperature_max, step=1, value=temperature_min)
selected_weather = st.sidebar.radio("Weather Condition:", weather_options)
selected_peak_season = st.sidebar.radio("Peak Season:", peak_season_options)
selected_holiday = st.sidebar.radio("Holiday:", holiday_options)

# Feature Engineering
df_encoded = df.copy()
df_encoded["airport"] = df_encoded["airport"].astype("category").cat.codes
df_encoded["season"] = df_encoded["season"].astype("category").cat.codes
df_encoded["weekday_weekend"] = df_encoded["weekday_weekend"].astype(int)
df_encoded["flight_type"] = df_encoded["flight_type"].astype(int)
df_encoded["weather_good"] = df_encoded["weather_good"].astype(int)
df_encoded["peak_season"] = df_encoded["peak_season"].astype(int)
df_encoded["holiday"] = df_encoded["holiday"].astype(int)

# Define features and target (removed total_flights, load_factor, economic_trend from user inputs)
features = [
    "airport", "season", "flight_type", "year", "weekday_weekend", "temperature",
    "total_flights", "load_factor", "weather_good", "economic_trend", "peak_season", "holiday"
]
target = "actual_footfall"

# Train the Model
if all(col in df_encoded.columns for col in features + [target]):
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display Model Performance
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")
    st.sidebar.write(f"R² Score: {r2:.2f}")
    st.sidebar.success("Model Trained Successfully ✅")

    # Predict for user input
    input_data = pd.DataFrame({
        "airport": [pd.Categorical([selected_airport], categories=df["airport"].unique()).codes[0]],
        "season": [pd.Categorical([selected_season], categories=df["season"].unique()).codes[0]],
        "flight_type": [0 if selected_flight_type == "Domestic" else 1],
        "year": [selected_year],
        "weekday_weekend": [0 if selected_weekday == "Weekday" else 1],
        "temperature": [selected_temperature],
        "total_flights": [mean_total_flights],  # Use mean value
        "load_factor": [mean_load_factor],      # Use mean value
        "weather_good": [1 if selected_weather == "Good" else 0],
        "economic_trend": [mean_economic_trend],  # Use mean value
        "peak_season": [1 if selected_peak_season == "Yes" else 0],
        "holiday": [1 if selected_holiday == "Yes" else 0]
    })

    predicted_footfall = model.predict(input_data)[0]

    # Display Predicted Footfall
    st.subheader("Footfall Prediction")
    st.write(f"### You can expect a footfall of {predicted_footfall:.0f}")

    # Bar Graph for Instant Analysis
    st.subheader("Quick Footfall Analysis")
    avg_footfall = df["actual_footfall"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["Average Footfall", "Predicted Footfall"], y=[avg_footfall, predicted_footfall], ax=ax)
    plt.ylabel("Footfall")
    plt.title("Average vs Predicted Footfall")
    st.pyplot(fig)

else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("Available columns:", df.columns.tolist())

# Footer
st.write("---")
st.write("Built")

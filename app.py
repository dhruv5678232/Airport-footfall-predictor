import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
from datetime import datetime

# Load dataset
file_url = "https://github.com/dhruv5678232/Airport-footfall-predictor/raw/main/Airport_Flight_Data_Cleaned.csv"
try:
    df = pd.read_csv(file_url)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Data Preprocessing
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Rename columns for consistency
df.rename(columns={
    "load_factor_(%)": "load_factor",
    "is_weekend": "weekday_weekend"
}, inplace=True)

# Enhanced flight_type logic (assuming dataset has both domestic and international data)
# If dataset has 'domestic_flights' and 'international_flights', use them
if 'domestic_flights' in df.columns and 'international_flights' in df.columns:
    df['flight_type'] = np.where(df['domestic_flights'] > df['international_flights'], 0, 1)  # 0: Domestic, 1: International
else:
    df['flight_type'] = (df['domestic_flights'] == 0).astype(int)  # Fallback logic

# Extract unique values for UI inputs
airports = df["airport"].dropna().unique().tolist()
seasons = df["season"].dropna().unique().tolist()
flight_types = ["Domestic", "International"]
weekday_options = ["Weekday", "Weekend"]
weather_options = ["Bad", "Good"]
temperature_min, temperature_max = int(df["temperature"].min()), int(df["temperature"].max())
peak_season_options = ["No", "Yes"]
holiday_options = ["No", "Yes"]
passenger_classes = ["Economy", "Premium Economy", "Business Class"]

# Debug and ensure 'year' column exists
if "year" not in df.columns:
    df['year'] = df['date'].dt.year  # Extract year from date if missing
df["year"] = pd.to_numeric(df["year"], errors='coerce')
df = df.dropna(subset=["year"])
years = sorted(df["year"].unique().tolist())

# Compute mean values for removed features
mean_total_flights = int(df["total_flights"].mean())
mean_load_factor = df["load_factor"].mean()
mean_economic_trend = df["economic_trend"].mean()

# Streamlit UI
st.title("Airport Footfall Prediction and Analysis")
st.sidebar.header("Input Parameters for Prediction")

# User inputs
selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)
if len(years) == 1:
    selected_year = int(years[0])
    st.sidebar.write(f"Year: {selected_year} (Only one year available)")
else:
    selected_year = st.sidebar.slider("Select Year:", min_value=int(min(years)), max_value=int(max(years)), step=1)
selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)
selected_temperature = st.sidebar.slider("Select Temperature (°C):", min_value=temperature_min, max_value=temperature_max, step26 = st.sidebar.slider("Select Temperature (°C):", min_value=temperature_min, max_value=temperature_max, step=1, value=temperature_min)
selected_weather = st.sidebar.radio("Weather Condition:", weather_options)
selected_peak_season = st.sidebar.radio("Peak Season:", peak_season_options)
selected_holiday = st.sidebar.radio("Holiday:", holiday_options)
selected_passenger_class = st.sidebar.selectbox("Select Passenger Class:", passenger_classes)

# Feature Engineering
df_encoded = df.copy()
df_encoded["airport"] = df_encoded["airport"].astype("category").cat.codes
df_encoded["season"] = df_encoded["season"].astype("category").cat.codes
df_encoded["weekday_weekend"] = df_encoded["weekday_weekend"].astype(int)
df_encoded["flight_type"] = df_encoded["flight_type"].astype(int)
df_encoded["weather_good"] = df_encoded["weather_good"].astype(int)
df_encoded["peak_season"] = df_encoded["peak_season"].astype(int)
df_encoded["holiday"] = df_encoded["holiday"].astype(int)

# Define features and target
features = [
    "airport", "season", "flight_type", "year", "weekday_weekend", "temperature",
    "total_flights", "load_factor", "weather_good", "economic_trend", "peak_season", "holiday"
]
target = "actual_footfall"

# Train the Model
if all(col in df_encoded.columns for col in features + [target]):
    X = df_encoded[features]
    y = df_encoded[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict for user input with distinct domestic/international logic
    input_data = pd.DataFrame({
        "airport": [pd.Categorical([selected_airport], categories=df["airport"].unique()).codes[0]],
        "season": [pd.Categorical([selected_season], categories=df["season"].unique()).codes[0]],
        "flight_type": [0 if selected_flight_type == "Domestic" else 1],
        "year": [selected_year],
        "weekday_weekend": [0 if selected_weekday == "Weekday" else 1],
        "temperature": [selected_temperature],
        "total_flights": [mean_total_flights],
        "load_factor": [mean_load_factor],
        "weather_good": [1 if selected_weather == "Good" else 0],
        "economic_trend": [mean_economic_trend],
        "peak_season": [1 if selected_peak_season == "Yes" else 0],
        "holiday": [1 if selected_holiday == "Yes" else 0]
    })

    predicted_footfall = model.predict(input_data)[0]
    predictions = np.array([tree.predict(input_data) for tree in model.estimators_])
    prediction_std = np.std(predictions)
    confidence_interval = 1.96 * prediction_std

    # Apply adjustments
    if selected_weekday == "Weekend":
        predicted_footfall *= 1.15
    seasonality_multipliers = {"Winter": 0.95, "Summer": 1.10, "Monsoon": 0.90}
    predicted_footfall *= seasonality_multipliers.get(selected_season, 1.0)
    # Differentiate by flight type
    flight_type_multiplier = 1.0 if selected_flight_type == "Domestic" else 1.25  # 25% higher for international
    predicted_footfall *= flight_type_multiplier

    st.subheader("Footfall Prediction")
    st.write(f"### You can expect a footfall of {predicted_footfall:,.0f} ± {confidence_interval:,.0f}")

    # Enhanced Revenue Calculation
    base_fare = 77.50  # USD, Economy Domestic
    fare_multipliers = {"Economy": 1.0, "Premium Economy": 1.5, "Business Class": 3.0}
    flight_fare_adjustment = 1.0 if selected_flight_type == "Domestic" else 1.5  # International fares 50% higher
    class_distribution = {"Economy": 0.7, "Premium Economy": 0.2, "Business Class": 0.1}
    
    selected_fare_multiplier = fare_multipliers[selected_passenger_class]
    weighted_fare = 0
    for class_type, proportion in class_distribution.items():
        fare = base_fare * fare_multipliers[class_type] * flight_fare_adjustment
        weighted_fare += fare * proportion
    adjusted_fare = weighted_fare * (1 + (fare_multipliers[selected_passenger_class] - 1) * 0.3)

    exchange_rate = 83
    revenue_usd = predicted_footfall * adjusted_fare
    revenue_inr = revenue_usd * exchange_rate

    if revenue_inr > 10000000:
        revenue_crores = revenue_inr / 10000000
        st.write(f"### Estimated Daily Revenue: {revenue_crores:,.2f} Crores")
    else:
        st.write(f"### Estimated Daily Revenue: ₹{revenue_inr:,.2f}")

    # Rest of the code (visualizations, future predictions, etc.) remains largely unchanged
    # Bar Graph
    st.subheader("Quick Footfall Analysis")
    avg_footfall = df["actual_footfall"].mean()
    fig = px.bar(x=["Average Footfall", "Predicted Footfall"], y=[avg_footfall, predicted_footfall],
                 labels={"x": "", "y": "Footfall"}, title="Average vs Predicted Footfall")
    st.plotly_chart(fig)

    # Future Footfall Predictions
    st.subheader("Future Footfall Predictions (2024-2035)")
    future_year = st.slider("Select a future year to predict footfall:", min_value=2025, max_value=2035, step=1, value=2030)
    base_footfall = predicted_footfall
    growth_rate = 0.038
    years_range = range(2024, future_year + 1)
    future_footfalls = [base_footfall * (1 + growth_rate) ** (year - 2024) for year in years_range]
    future_df = pd.DataFrame({"Year": list(years_range), "Predicted Footfall": future_footfalls})
    fig = px.line(future_df, x="Year", y="Predicted Footfall", markers=True, title=f"Footfall Trend from 2024 to {future_year}")
    st.plotly_chart(fig)
    st.write(f"### Expected Footfall in {future_year}: {future_footfalls[-1]:,.0f}")

    # Sensitivity Analysis and Export sections can remain as is unless further adjustments are needed
else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("Available columns:", df.columns.tolist())

st.write("---")
st.write("Built")

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
df = pd.read_csv(file_url)

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
years = sorted(df["year"].dropna().unique().tolist())
weather_options = ["Bad", "Good"]
economic_trend_min, economic_trend_max = df["economic_trend"].min(), df["economic_trend"].max()
temperature_min, temperature_max = int(df["temperature"].min()), int(df["temperature"].max())
total_flights_min, total_flights_max = int(df["total_flights"].min()), int(df["total_flights"].max())
load_factor_min, load_factor_max = df["load_factor"].min(), df["load_factor"].max()
peak_season_options = ["No", "Yes"]
holiday_options = ["No", "Yes"]

# Streamlit UI
st.title("Airport Footfall Prediction and Analysis")
st.sidebar.header("Input Parameters for Prediction")

# User inputs for all factors
selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)

# Handle the year selection (fix for StreamlitAPIException)
if len(years) == 1:
    selected_year = years[0]
    st.sidebar.write(f"Year: {selected_year} (Only one year available in the dataset)")
else:
    selected_year = st.sidebar.slider("Select Year:", min_value=int(min(years)), max_value=int(max(years)), step=1)

selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)
selected_temperature = st.sidebar.slider("Select Temperature (°C):", min_value=temperature_min, max_value=temperature_max, step=1, value=temperature_min)
selected_total_flights = st.sidebar.slider("Select Total Flights:", min_value=total_flights_min, max_value=total_flights_max, step=1, value=total_flights_min)
selected_load_factor = st.sidebar.slider("Select Load Factor (%):", min_value=float(load_factor_min), max_value=float(load_factor_max), step=0.1, value=float(load_factor_min))
selected_weather = st.sidebar.radio("Weather Condition:", weather_options)
selected_economic_trend = st.sidebar.slider("Select Economic Trend:", min_value=float(economic_trend_min), max_value=float(economic_trend_max), step=0.01, value=float(economic_trend_min))
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

    # Visualization of Actual vs Predicted Footfall
    st.subheader("Actual vs Predicted Footfall Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_encoded["predicted_footfall"] = model.predict(df_encoded[features])
    sns.lineplot(x=df["date"], y=df["actual_footfall"], marker="o", label="Actual Footfall", ax=ax)
    sns.lineplot(x=df["date"], y=df_encoded["predicted_footfall"], marker="s", label="Predicted Footfall", ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Footfall")
    plt.legend()
    st.pyplot(fig)

    # Predict for user input
    input_data = pd.DataFrame({
        "airport": [pd.Categorical([selected_airport], categories=df["airport"].unique()).codes[0]],
        "season": [pd.Categorical([selected_season], categories=df["season"].unique()).codes[0]],
        "flight_type": [0 if selected_flight_type == "Domestic" else 1],
        "year": [selected_year],
        "weekday_weekend": [0 if selected_weekday == "Weekday" else 1],
        "temperature": [selected_temperature],
        "total_flights": [selected_total_flights],
        "load_factor": [selected_load_factor],
        "weather_good": [1 if selected_weather == "Good" else 0],
        "economic_trend": [selected_economic_trend],
        "peak_season": [1 if selected_peak_season == "Yes" else 0],
        "holiday": [1 if selected_holiday == "Yes" else 0]
    })

    predicted_footfall = model.predict(input_data)[0]
    st.write(f"### Predicted Footfall: {predicted_footfall:.0f}")

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.write(feature_importance)

    # Analysis Section
    st.subheader("Analysis of Factors Impacting Footfall")

    # 1. Temperature Analysis
    st.write("#### 1. Temperature Impact")
    temp_bins = pd.cut(df["temperature"], bins=[0, 15, 25, 35, 45], labels=["Low (<15°C)", "Medium (15-25°C)", "High (25-35°C)", "Very High (>35°C)"])
    temp_analysis = df.groupby(temp_bins)["actual_footfall"].mean().round(0)
    st.write("Average Footfall by Temperature Range:")
    st.write(temp_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=temp_bins, y=df["actual_footfall"], ax=ax)
    plt.xlabel("Temperature Range")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 2. Total Flights Analysis
    st.write("#### 2. Total Flights Impact")
    flights_bins = pd.cut(df["total_flights"], bins=[0, 300, 450, 600], labels=["Low (<300)", "Medium (300-450)", "High (>450)"])
    flights_analysis = df.groupby(flights_bins)["actual_footfall"].mean().round(0)
    st.write("Average Footfall by Total Flights Range:")
    st.write(flights_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=flights_bins, y=df["actual_footfall"], ax=ax)
    plt.xlabel("Total Flights Range")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 3. Domestic Flights Analysis
    st.write("#### 3. Domestic Flights Impact")
    domestic_bins = pd.cut(df["domestic_flights"], bins=[0, 200, 350, 500], labels=["Low (<200)", "Medium (200-350)", "High (>350)"])
    domestic_analysis = df.groupby(domestic_bins)["actual_footfall"].mean().round(0)
    st.write("Average Footfall by Domestic Flights Range:")
    st.write(domestic_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=domestic_bins, y=df["actual_footfall"], ax=ax)
    plt.xlabel("Domestic Flights Range")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 4. International Flights Analysis
    st.write("#### 4. International Flights Impact")
    international_bins = pd.cut(df["international_flights"], bins=[0, 70, 120, 210], labels=["Low (<70)", "Medium (70-120)", "High (>120)"])
    international_analysis = df.groupby(international_bins)["actual_footfall"].mean().round(0)
    st.write("Average Footfall by International Flights Range:")
    st.write(international_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=international_bins, y=df["actual_footfall"], ax=ax)
    plt.xlabel("International Flights Range")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 5. Load Factor Analysis
    st.write("#### 5. Load Factor Impact")
    load_factor_bins = pd.cut(df["load_factor"], bins=[0, 75, 85, 100], labels=["Low (<75%)", "Medium (75-85%)", "High (>85%)"])
    load_factor_analysis = df.groupby(load_factor_bins)["actual_footfall"].mean().round(0)
    st.write("Average Footfall by Load Factor Range:")
    st.write(load_factor_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=load_factor_bins, y=df["actual_footfall"], ax=ax)
    plt.xlabel("Load Factor Range")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 6. Weather Good Analysis
    st.write("#### 6. Weather Impact")
    weather_analysis = df.groupby("weather_good")["actual_footfall"].mean().round(0)
    weather_analysis.index = ["Bad Weather", "Good Weather"]
    st.write("Average Footfall by Weather Condition:")
    st.write(weather_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="weather_good", y="actual_footfall", data=df, ax=ax)
    plt.xlabel("Weather Condition (0 = Bad, 1 = Good)")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 7. Economic Trend Analysis
    st.write("#### 7. Economic Trend Impact")
    economic_bins = pd.cut(df["economic_trend"], bins=[0.7, 0.9, 1.1, 1.3], labels=["Low (<0.9)", "Medium (0.9-1.1)", "High (>1.1)"])
    economic_analysis = df.groupby(economic_bins)["actual_footfall"].mean().round(0)
    st.write("Average Footfall by Economic Trend Range:")
    st.write(economic_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=economic_bins, y=df["actual_footfall"], ax=ax)
    plt.xlabel("Economic Trend Range")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 8. Weekday/Weekend Analysis
    st.write("#### 8. Weekday/Weekend Impact")
    weekday_analysis = df.groupby("weekday_weekend")["actual_footfall"].mean().round(0)
    weekday_analysis.index = ["Weekday", "Weekend"]
    st.write("Average Footfall by Weekday/Weekend:")
    st.write(weekday_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="weekday_weekend", y="actual_footfall", data=df, ax=ax)
    plt.xlabel("Flight Day (0 = Weekday, 1 = Weekend)")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 9. Peak Season Analysis
    st.write("#### 9. Peak Season Impact")
    peak_season_analysis = df.groupby("peak_season")["actual_footfall"].mean().round(0)
    peak_season_analysis.index = ["Not Peak Season", "Peak Season"]
    st.write("Average Footfall by Peak Season:")
    st.write(peak_season_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="peak_season", y="actual_footfall", data=df, ax=ax)
    plt.xlabel("Peak Season (0 = No, 1 = Yes)")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 10. Holiday Analysis
    st.write("#### 10. Holiday Impact")
    holiday_analysis = df.groupby("holiday")["actual_footfall"].mean().round(0)
    holiday_analysis.index = ["Not a Holiday", "Holiday"]
    st.write("Average Footfall by Holiday:")
    st.write(holiday_analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="holiday", y="actual_footfall", data=df, ax=ax)
    plt.xlabel("Holiday (0 = No, 1 = Yes)")
    plt.ylabel("Actual Footfall")
    st.pyplot(fig)

    # 11. Year Analysis (for completeness, though only one year exists)
    st.write("#### 11. Year Impact")
    year_analysis = df.groupby("year")["actual_footfall"].mean().round(0)
    st.write("Average Footfall by Year:")
    st.write(year_analysis)

else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("Available columns:", df.columns.tolist())

# Footer
st.write("---")
st.write("Built with ❤️ ")

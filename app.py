import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
try:
    df = pd.read_csv(file_url)
    df.columns = df.columns.str.lower()  # Normalize column names
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Convert 'date' column to datetime format
try:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows where date conversion failed
except Exception as e:
    st.error(f"Error processing dates: {e}")
    st.stop()

# Extracting values safely
airports = df["airport"].dropna().unique().tolist()
seasons = df["season"].dropna().unique().tolist()
flight_types = ["Domestic", "International"]
weekday_options = ["Weekday", "Weekend"]

# Extract years safely
years = df["date"].dt.year.dropna().unique().tolist()
if not years:
    st.error("No valid years found in the dataset.")
    st.stop()

# Streamlit UI
st.title("Airport Footfall Prediction")
st.sidebar.header("Input Parameters")

selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)
selected_year = st.sidebar.slider("Select Year:", min_value=int(min(years)), max_value=int(max(years)), step=1)
selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)

# Display selected inputs
st.write("### Selected Inputs")
st.write(f"Airport: {selected_airport}")
st.write(f"Season: {selected_season}")
st.write(f"Flight Type: {selected_flight_type}")
st.write(f"Year: {selected_year}")
st.write(f"Day Type: {selected_weekday}")

# Feature Engineering
df_encoded = df.copy()
df_encoded["airport"] = df_encoded["airport"].astype("category").cat.codes
df_encoded["season"] = df_encoded["season"].astype("category").cat.codes
df_encoded["is_weekend"] = df_encoded["is_weekend"].astype(int)  # Binary encoding
df_encoded["year"] = df_encoded["date"].dt.year
df_encoded["flight_type"] = df_encoded["domestic_flights"].apply(lambda x: 0 if x > 0 else 1)

# Select relevant features
features = ["airport", "season", "flight_type", "year", "is_weekend", "load_factor (%)", "total_flights"]
target = "actual_footfall"

# Ensure all required columns exist before training
if all(col in df_encoded.columns for col in features + [target]):
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display Performance
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")
    st.sidebar.write(f"R² Score: {r2:.2f}")
    st.sidebar.success("Model Trained Successfully ✅")

    # Visualization
    st.subheader("Data Visualization")
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
    airport_code = pd.Categorical([selected_airport], categories=df["airport"].unique()).codes[0]
    season_code = pd.Categorical([selected_season], categories=df["season"].unique()).codes[0]
    flight_type_code = 0 if selected_flight_type == "Domestic" else 1
    is_weekend_code = 0 if selected_weekday == "Weekday" else 1

    input_data = pd.DataFrame({
        "airport": [airport_code],
        "season": [season_code],
        "flight_type": [flight_type_code],
        "year": [selected_year],
        "is_weekend": [is_weekend_code],
        "load_factor (%)": [df["load_factor (%)"].mean()],  # Use mean as placeholder
        "total_flights": [df["total_flights"].mean()]       # Use mean as placeholder
    })

    predicted_footfall = model.predict(input_data)[0]
    st.write(f"### Predicted Footfall: {predicted_footfall:.0f}")

else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("Available columns:", df.columns.tolist())

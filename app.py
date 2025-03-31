import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "Airport_Flight_Data_Final_Updated.csv"
try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()  # Normalize column names
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Ensure date column exists and parse it
if "date" in df.columns:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates
    df['year'] = df['date'].dt.year.astype('Int64')  # Extract year as integer
else:
    st.error("Date column not found in the dataset.")
    st.stop()

# Ensure required columns exist
df.rename(columns={
    "load_factor (%)": "load_factor",
    "is_weekend": "weekday_weekend"
}, inplace=True)

# Create 'flight_type' column
df['flight_type'] = (df['domestic_flights'] == 0).astype(int)

# Extracting unique values safely
airports = df["airport"].dropna().unique().tolist()
seasons = df["season"].dropna().unique().tolist()
flight_types = ["Domestic", "International"]
weekday_options = ["Weekday", "Weekend"]
years = sorted(df["year"].dropna().unique().tolist())

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

# Feature Engineering
df_encoded = df.copy()
df_encoded["airport"] = df_encoded["airport"].astype("category").cat.codes
df_encoded["season"] = df_encoded["season"].astype("category").cat.codes
df_encoded["weekday_weekend"] = df_encoded["weekday_weekend"].astype(int)  # Ensure it's integer

df_encoded["flight_type"] = df_encoded["flight_type"].astype(int)

# Select relevant features
features = ["airport", "season", "flight_type", "year", "weekday_weekend", "load_factor", "total_flights"]
target = "actual_footfall"

# Ensure all required columns exist
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
    input_data = pd.DataFrame({
        "airport": [pd.Categorical([selected_airport], categories=df["airport"].unique()).codes[0]],
        "season": [pd.Categorical([selected_season], categories=df["season"].unique()).codes[0]],
        "flight_type": [0 if selected_flight_type == "Domestic" else 1],
        "year": [selected_year],
        "weekday_weekend": [0 if selected_weekday == "Weekday" else 1],
        "load_factor": [df["load_factor"].mean()],
        "total_flights": [df["total_flights"].mean()]
    })

    predicted_footfall = model.predict(input_data)[0]
    st.write(f"### Predicted Footfall: {predicted_footfall:.0f}")

else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("Available columns:", df.columns.tolist())

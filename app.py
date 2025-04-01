import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import io

# Load dataset
file_url = "https://github.com/dhruv5678232/Airport-footfall-predictor/raw/main/Airport_Flight_Data_Cleaned.csv"
df = pd.read_csv(file_url)

df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

df.rename(columns={"load_factor_(%)": "load_factor", "is_weekend": "weekday_weekend"}, inplace=True)
df['flight_type'] = df['domestic_flights'].apply(lambda x: "International" if x == 0 else "Domestic")

airports = df["airport"].dropna().unique().tolist()
seasons = df["season"].dropna().unique().tolist()
flight_types = ["Domestic", "International"]
years = sorted(df["year"].dropna().unique().tolist())

df_encoded = df.copy()
df_encoded["airport"] = df_encoded["airport"].astype("category").cat.codes
df_encoded["season"] = df_encoded["season"].astype("category").cat.codes
df_encoded["flight_type"] = df_encoded["flight_type"].map({"Domestic": 0, "International": 1})
df_encoded["weekday_weekend"] = df_encoded["weekday_weekend"].astype(int)

df_encoded = df_encoded.dropna(subset=["year", "actual_footfall"])
features = ["airport", "season", "flight_type", "year", "weekday_weekend", "temperature", "total_flights", "load_factor"]
target = "actual_footfall"

X = df_encoded[features]
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Airport Footfall Prediction")
selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)
selected_year = st.sidebar.slider("Select Year:", min_value=int(min(years)), max_value=int(max(years)), step=1)

flight_type_encoded = 1 if selected_flight_type == "International" else 0
input_data = pd.DataFrame({
    "airport": [pd.Categorical([selected_airport], categories=df["airport"].unique()).codes[0]],
    "season": [pd.Categorical([selected_season], categories=df["season"].unique()).codes[0]],
    "flight_type": [flight_type_encoded],
    "year": [selected_year],
    "weekday_weekend": [0],
    "temperature": [20],
    "total_flights": [df["total_flights"].mean()],
    "load_factor": [df["load_factor"].mean()]
})

predicted_footfall = model.predict(input_data)[0]
st.subheader(f"Predicted Footfall for {selected_flight_type}: {predicted_footfall:,.0f}")

# Export Predictions
export_data = pd.DataFrame({"Predicted Footfall": [predicted_footfall], "Flight Type": [selected_flight_type]})
csv_buffer = io.StringIO()
export_data.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Predictions as CSV",
    data=csv_buffer.getvalue(),
    file_name=f"footfall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

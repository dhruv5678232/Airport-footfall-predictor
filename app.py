import streamlit as st
import pandas as pd

# Load dataset from GitHub (ensure it's in raw format)
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
try:
    df = pd.read_csv(file_url)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Ensure 'Year' column is numeric and drop NaN values
df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

# Extract unique values
airports = df["Airport"].unique().tolist()
seasons = ["Summer", "Monsoon", "Winter"]
flight_types = ["Domestic", "International"]

# Ensure a valid range for the year selection
if df["Year"].empty:
    years = [2023]  # Default year if no valid data
else:
    years = sorted(df["Year"].unique().tolist())

weekday_options = ["Weekday", "Weekend"]

# Streamlit UI
st.title("Airport Footfall Prediction")
st.sidebar.header("Input Parameters")

selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)
selected_year = st.sidebar.slider("Select Year:", min_value=min(years), max_value=max(years), step=1)
selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)

# Display selected inputs
st.write("### Selected Inputs")
st.write(f"**Airport:** {selected_airport}")
st.write(f"**Season:** {selected_season}")
st.write(f"**Flight Type:** {selected_flight_type}")
st.write(f"**Year:** {selected_year}")
st.write(f"**Day Type:** {selected_weekday}")

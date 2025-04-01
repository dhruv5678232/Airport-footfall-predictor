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
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    st.error("The 'reportlab' library is not installed. Please add 'reportlab' to your requirements.txt and redeploy the app.")
    st.stop()

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
weather_options = ["Bad", "Good"]
temperature_min, temperature_max = int(df["temperature"].min()), int(df["temperature"].max())
peak_season_options = ["No", "Yes"]
holiday_options = ["No", "Yes"]
passenger_classes = ["Economy", "Premium Economy", "Business Class"]

# Debug the 'year' column
if "year" not in df.columns:
    st.error("The 'year' column is missing from the dataset.")
    st.stop()

# Ensure 'year' column is numeric and handle missing values
df["year"] = pd.to_numeric(df["year"], errors='coerce')
df = df.dropna(subset=["year"])  # Remove rows with invalid years
years = sorted(df["year"].unique().tolist())

# Compute mean values for removed features
mean_total_flights = int(df["total_flights"].mean())
mean_load_factor = df["load_factor"].mean()
mean_economic_trend = df["economic_trend"].mean()

# Streamlit UI
st.title("Airport Footfall Prediction and Analysis")
st.sidebar.header("Input Parameters for Prediction")

# User inputs
selected_airport = st.sidebar.selectbox("Select Airport

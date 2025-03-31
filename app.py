import streamlit as st
import pandas as pd
import numpy as np

# Load dataset (Fixed CSV Issue)
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
try:
    df = pd.read_csv(file_url)

    # Extract unique values
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

    # ✅ Feature Engineering (New Additions)
    st.sidebar.header("Feature Engineering")

    # Handle missing values (fill with median)
    for col in ["Load_factor", "Predicted_footfall", "Actual_footfall"]:
        if col in df:
            df[col].fillna(df[col].median(), inplace=True)

    # Compute seasonal average footfall per airport
    if "Predicted_footfall" in df and "Season" in df:
        seasonal_footfall = df.groupby(["Airport", "Season"])["Predicted_footfall"].mean().reset_index()
        st.sidebar.write("Seasonal Avg Footfall Computed ✅")

    # Encode categorical features for ML
    categorical_cols = ["Airport", "Season", "Flight Type", "Weekday/Weekend"]
    for col in categorical_cols:
        if col in df:
            df[col] = df[col].astype("category").cat.codes  # Converts categories to numbers

    # Extract historical footfall trends
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year  # Ensure Year column exists
        df["Monthly_Trend"] = df.groupby(["Airport", df["Date"].dt.month])["Predicted_footfall"].transform("mean")

        st.sidebar.write("Historical Footfall Trends Extracted ✅")

except Exception as e:
    st.error(f"Error loading dataset: {e}")

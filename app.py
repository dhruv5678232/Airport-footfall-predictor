import streamlit as st
import pandas as pd
import numpy as np

# Load dataset (Fixed CSV Issue)
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
try:
    df = pd.read_csv(file_url)

    # ✅ Convert all column names to lowercase (Fix inconsistent column names)
    df.columns = df.columns.str.lower()

    # Extract unique values
    airports = df["airport"].dropna().unique().tolist()
    seasons = df["season"].dropna().unique().tolist() if "season" in df else ["Summer", "Monsoon", "Winter"]
    flight_types = ["Domestic", "International"]
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df else []
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
    for col in ["load_factor", "predicted_footfall", "actual_footfall"]:
        if col in df:
            df[col].fillna(df[col].median(), inplace=True)

    # Compute seasonal average footfall per airport
    if "predicted_footfall" in df and "season" in df:
        seasonal_footfall = df.groupby(["airport", "season"])["predicted_footfall"].mean().reset_index()
        st.sidebar.write("Seasonal Avg Footfall Computed ✅")

    # Encode categorical features for ML
    categorical_cols = ["airport", "season", "flight type", "weekday/weekend"]
    for col in categorical_cols:
        if col in df:
            df[col] = df[col].astype("category").cat.codes  # Converts categories to numbers

    # ✅ Fix Date Parsing Issue
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")  # Fix date format
        df["year"] = df["date"].dt.year  # Extract Year from Date

        # Extract historical trends
        df["monthly_trend"] = df.groupby(["airport", df["date"].dt.month])["predicted_footfall"].transform("mean")

        st.sidebar.write("Historical Footfall Trends Extracted ✅")

except Exception as e:
    st.error(f"Error loading dataset: {e}")

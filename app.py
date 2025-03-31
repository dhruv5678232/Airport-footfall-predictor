import streamlit as st
import pandas as pd

# Load dataset from GitHub (ensure it's in raw format)
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
try:
    df = pd.read_csv(file_url)
    
    # Ensure column names are properly formatted
    df.columns = df.columns.str.strip()
    
    # Extract unique values safely
    airports = df["Airport"].dropna().unique().tolist() if "Airport" in df else []
    seasons = df["Season"].dropna().unique().tolist() if "Season" in df else ["Summer", "Monsoon", "Winter"]
    flight_types = ["Domestic", "International"]
    years = sorted(df["Year"].dropna().unique().tolist()) if "Year" in df else []
    weekday_options = ["Weekday", "Weekend"]
    
    # Handle case where dataset is empty or missing values
    if not years:
        years = list(range(2020, 2031))  # Default range if no valid data
    
    # Streamlit UI
    st.title("Airport Footfall Prediction")
    st.sidebar.header("Input Parameters")
    
    selected_airport = st.sidebar.selectbox("Select Airport:", airports if airports else ["No data available"])
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
except Exception as e:
    st.error(f"Error loading dataset: {e}")

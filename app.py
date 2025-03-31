import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (using the uploaded data as a placeholder since GitHub URL isn’t accessible here)
# In practice, replace this with the GitHub URL: "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
data = """
<Insert the full CSV data you provided here>
"""
df = pd.read_csv(pd.compat.StringIO(data))  # For this example, using StringIO; replace with pd.read_csv(file_url) in production

# Convert all column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Extract unique values from the dataset
airports = df["airport"].dropna().unique().tolist()
seasons = df["season"].dropna().unique().tolist()
flight_types = ["Domestic", "International"]  # Inferred from "domestic_flights" and "international_flights"
years = pd.to_datetime(df["date"]).dt.year.unique().tolist()  # Extract year from "date" column
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
st.write(f"Airport: {selected_airport}")
st.write(f"Season: {selected_season}")
st.write(f"Flight Type: {selected_flight_type}")
st.write(f"Year: {selected_year}")
st.write(f"Day Type: {selected_weekday}")

# Feature Engineering
# Convert categorical columns to numeric codes
df["airport"] = df["airport"].astype("category").cat.codes
df["season"] = df["season"].astype("category").cat.codes
df["is_weekend"] = df["is_weekend"].astype(int)  # Already binary (0 or 1)
df["year"] = pd.to_datetime(df["date"]).dt.year  # Extract year from date

# Create a "flight_type" column based on user selection (simplified assumption)
df["flight_type"] = 0 if selected_flight_type == "Domestic" else 1 остро

# Select relevant features (adjusted to match dataset)
features = ["airport", "season", "flight_type", "year", "is_weekend", "load_factor (%)", "total_flights"]
target = "actual_footfall"

# Ensure all required columns exist
if all(col in df for col in features + [target]):
    # Train-test split (80-20)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display Model Performance
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")
    st.sidebar.write(f"R² Score: {r2:.2f}")

    st.sidebar.success("Model Trained Successfully ✅")

    # Visualization
    st.subheader("Data Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Add predictions to the dataframe for plotting
    df["predicted_footfall"] = model.predict(df[features])
    
    # Plot actual vs predicted footfall over time
    sns.lineplot(x=df["date"], y=df["actual_footfall"], marker="o", label="Actual Footfall", ax=ax)
    sns.lineplot(x=df["date"], y=df["predicted_footfall"], marker="s", label="Predicted Footfall", ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Footfall")
    plt.legend()
    st.pyplot(fig)

    # Predict footfall for user input
    input_data = pd.DataFrame({
        "airport": [df["airport"].iloc[0]],  # Use first airport code as placeholder
        "season": [df["season"].iloc[0]],    # Use first season code as placeholder
        "flight_type": [0 if selected_flight_type == "Domestic" else 1],
        "year": [selected_year],
        "is_weekend": [0 if selected_weekday == "Weekday" else 1],
        "load_factor (%)": [df["load_factor (%)"].mean()],  # Use mean as placeholder
        "total_flights": [df["total_flights"].mean()]       # Use mean as placeholder
    })
    predicted_footfall = model.predict(input_data)[0]
    st.write(f"### Predicted Footfall: {predicted_footfall:.0f}")

else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("Available columns:", df.columns.tolist())

# Handle exceptions
try:
    pass  # Main logic is already inside the if block
except Exception as e:
    st.error(f"Error processing dataset: {e}")

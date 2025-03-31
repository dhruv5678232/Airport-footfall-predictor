import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "Airport_Flight_Data_Final_Updated.csv"
df = pd.read_csv(file_path)

# Feature Engineering
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Encode categorical features
label_encoders = {}
categorical_cols = ["Season", "Weather_Good", "Economic_Trend", "Airport"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for future inverse transform

# Prepare dataset for ML
X = df[["Year", "Month", "Airport", "Season", "Weather_Good", "Economic_Trend", "Total_Flights"]]
y = df["Actual_Footfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Airport Footfall Predictor", layout="wide")
st.title("✈️ Airport Footfall Prediction")

# Future Footfall Prediction
st.subheader("🔮 Predict Future Airport Footfall")
future_year = st.slider("Select Future Year:", min_value=df["Year"].max() + 1, max_value=df["Year"].max() + 10, step=1)
future_month = st.slider("Select Month:", min_value=1, max_value=12, step=1)
departure_airport = st.selectbox("Select Departure Airport:", df["Airport"].unique())
arrival_airport = st.selectbox("Select Arrival Airport:", df["Airport"].unique())

# Predict button
if st.button("🚀 Predict"):
    # Encode input values
    dep_airport_encoded = label_encoders["Airport"].transform([departure_airport])[0]
    arr_airport_encoded = label_encoders["Airport"].transform([arrival_airport])[0]
    
    # Prepare input for prediction
    input_data = np.array([[future_year, future_month, dep_airport_encoded, arr_airport_encoded, 1, 1, 100]])
    
    # Predict Footfall
    predicted_footfall = model.predict(input_data)[0]
    
    # Display Prediction
    st.subheader(f"📊 Predicted Footfall: **{int(predicted_footfall)} passengers**")
    
    # Visualization
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=df["Year"], y=df["Actual_Footfall"], marker="o", label="Past Data")
    plt.axvline(x=future_year, color="r", linestyle="--", label="Prediction Point")
    plt.scatter(future_year, predicted_footfall, color="red", s=100, label="Predicted Footfall")
    plt.xlabel("Year")
    plt.ylabel("Passenger Footfall")
    plt.legend()
    st.pyplot(plt)

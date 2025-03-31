import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load dataset from GitHub
file_url = "https://raw.githubusercontent.com/dhruv5678232/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"

try:
    df = pd.read_csv(file_url)
    
    # ✅ Convert all column names to lowercase (Fix inconsistent column names)
    df.columns = df.columns.str.lower()
    
    # ✅ Check for missing required columns
    required_columns = {"airport", "season", "flight type", "year", "weekday/weekend", "load_factor", "actual_footfall"}
    missing_cols = required_columns - set(df.columns)
    
    if missing_cols:
        st.sidebar.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.stop()  # Stop execution if columns are missing

    # ✅ Extract unique values for UI
    airports = df["airport"].dropna().unique().tolist()
    seasons = df["season"].dropna().unique().tolist()
    flight_types = ["Domestic", "International"]
    years = sorted(df["year"].dropna().unique().tolist())
    weekday_options = ["Weekday", "Weekend"]
    
    # ✅ Streamlit UI
    st.title("Airport Footfall Prediction")
    st.sidebar.header("Input Parameters")
    
    selected_airport = st.sidebar.selectbox("Select Airport:", airports)
    selected_season = st.sidebar.selectbox("Select Season:", seasons)
    selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)
    selected_year = st.sidebar.slider("Select Year:", min_value=min(years), max_value=max(years), step=1)
    selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)
    
    # ✅ Display selected inputs
    st.subheader("Selected Inputs")
    st.write(f"**Airport:** {selected_airport}")
    st.write(f"**Season:** {selected_season}")
    st.write(f"**Flight Type:** {selected_flight_type}")
    st.write(f"**Year:** {selected_year}")
    st.write(f"**Day Type:** {selected_weekday}")

    # ✅ Feature Engineering
    categorical_cols = ["airport", "season", "flight type", "weekday/weekend"]
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes  # Convert categories to numbers
    
    # ✅ Select Relevant Features & Target
    features = ["airport", "season", "flight type", "year", "weekday/weekend", "load_factor"]
    target = "actual_footfall"

    # ✅ Train-Test Split (80-20)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Make Predictions
    y_pred = model.predict(X_test)

    # ✅ Evaluate Model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # ✅ Display Model Performance
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"📉 **MAE:** {mae:.2f}")
    st.sidebar.write(f"📉 **RMSE:** {rmse:.2f}")
    st.sidebar.write(f"📈 **R² Score:** {r2:.2f}")

    st.sidebar.success("✅ Model Trained Successfully")

    # ✅ Data Visualization
    st.subheader("📊 Footfall Trends")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=df["year"], y=df["actual_footfall"], marker="o", label="Actual Footfall", ax=ax)
    sns.lineplot(x=df["year"], y=df["predicted_footfall"], marker="s", label="Predicted Footfall", ax=ax)
    plt.xlabel("Year")
    plt.ylabel("Footfall")
    plt.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ Error loading dataset: {e}")

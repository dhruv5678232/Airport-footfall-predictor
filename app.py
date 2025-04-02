import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import BytesIO, StringIO  # Updated import to include StringIO
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
st.title("AeroPredict Solutions - Airport Footfall Prediction and Analysis")
st.sidebar.header("Input Parameters for Prediction")

# User inputs
selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)

# Handle the year selection
if len(years) == 0:
    st.sidebar.error("No valid years found in the dataset.")
    st.stop()
elif len(years) == 1:
    selected_year = int(years[0])  # Ensure integer
    st.sidebar.write(f"Year: {selected_year} (Only one year available in the dataset)")
else:
    selected_year = st.sidebar.slider("Select Year:", min_value=int(min(years)), max_value=int(max(years)), step=1)

selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)
selected_temperature = st.sidebar.slider("Select Temperature (°C):", min_value=temperature_min, max_value=temperature_max, step=1, value=temperature_min)
selected_weather = st.sidebar.radio("Weather Condition:", weather_options)
selected_peak_season = st.sidebar.radio("Peak Season:", peak_season_options)
selected_holiday = st.sidebar.radio("Holiday:", holiday_options)
selected_passenger_class = st.sidebar.selectbox("Select Passenger Class:", passenger_classes)

# Feature Engineering
df_encoded = df.copy()
df_encoded["airport"] = df_encoded["airport"].astype("category").cat.codes
df_encoded["season"] = df_encoded["season"].astype("category").cat.codes
df_encoded["weekday_weekend"] = df_encoded["weekday_weekend"].astype(int)
df_encoded["flight_type"] = df_encoded["flight_type"].astype(int)
df_encoded["weather_good"] = df_encoded["weather_good"].astype(int)
df_encoded["peak_season"] = df_encoded["peak_season"].astype(int)
df_encoded["holiday"] = df_encoded["holiday"].astype(int)

# Define features and target
features = [
    "airport", "season", "flight_type", "year", "weekday_weekend", "temperature",
    "total_flights", "load_factor", "weather_good", "economic_trend", "peak_season", "holiday"
]
target = "actual_footfall"

# Train the Model
if all(col in df_encoded.columns for col in features + [target]):
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict for user input
    input_data = pd.DataFrame({
        "airport": [pd.Categorical([selected_airport], categories=df["airport"].unique()).codes[0]],
        "season": [pd.Categorical([selected_season], categories=df["season"].unique()).codes[0]],
        "flight_type": [0 if selected_flight_type == "Domestic" else 1],
        "year": [selected_year],
        "weekday_weekend": [0 if selected_weekday == "Weekday" else 1],
        "temperature": [selected_temperature],
        "total_flights": [mean_total_flights],
        "load_factor": [mean_load_factor],
        "weather_good": [1 if selected_weather == "Good" else 0],
        "economic_trend": [mean_economic_trend],
        "peak_season": [1 if selected_peak_season == "Yes" else 0],
        "holiday": [1 if selected_holiday == "Yes" else 0]
    })

    # Predict footfall with confidence interval
    predicted_footfall = model.predict(input_data)[0]
    
    # Estimate confidence interval (approximation using standard deviation of predictions)
    predictions = np.array([tree.predict(input_data) for tree in model.estimators_])
    prediction_std = np.std(predictions)
    confidence_interval = 1.96 * prediction_std  # 95% confidence interval

    # Apply weekend multiplier (15% increase for weekends)
    if selected_weekday == "Weekend":
        predicted_footfall *= 1.15

    # Apply seasonality adjustment
    seasonality_multipliers = {"Winter": 0.95, "Summer": 1.10, "Monsoon": 0.90}
    predicted_footfall *= seasonality_multipliers.get(selected_season, 1.0)

    # Display Predicted Footfall with Confidence Interval
    st.subheader("Footfall Prediction")
    st.write(f"### You can expect a footfall of {predicted_footfall:,.0f} (plus or minus {confidence_interval:,.0f})")

    # Calculate Revenue
    base_fare = 77.50  # USD, one-way fare for Economy
    fare_multipliers = {"Economy": 1.0, "Premium Economy": 1.5, "Business Class": 3.0}
    class_distribution = {"Economy": 0.7, "Premium Economy": 0.2, "Business Class": 0.1}
    
    # Adjust fare based on selected class
    selected_fare_multiplier = fare_multipliers[selected_passenger_class]
    weighted_fare = 0
    for class_type, proportion in class_distribution.items():
        fare = base_fare * fare_multipliers[class_type]
        weighted_fare += fare * proportion
    adjusted_fare = weighted_fare * (1 + (fare_multipliers[selected_passenger_class] - 1) * 0.3)  # Adjust based on selected class

    # Convert to INR
    exchange_rate = 83  # 1 USD = 83 INR
    revenue_usd = predicted_footfall * adjusted_fare
    revenue_inr = revenue_usd * exchange_rate

    # Display revenue in INR (with crores if > 1,00,00,000)
    if revenue_inr > 10000000:
        revenue_crores = revenue_inr / 10000000
        st.write(f"### Estimated Daily Revenue: {revenue_crores:,.2f} Crores")
    else:
        st.write(f"### Estimated Daily Revenue: ₹{revenue_inr:,.2f}")

    # Calculate average revenue for comparison
    avg_footfall = df["actual_footfall"].mean()
    avg_revenue_usd = avg_footfall * adjusted_fare  # Using the same adjusted_fare as for predicted revenue
    avg_revenue_inr = avg_revenue_usd * exchange_rate

    # Display a bar graph comparing average vs predicted revenue
    st.subheader("Revenue Comparison")
    fig = px.bar(
        x=["Average Revenue", "Predicted Revenue"],
        y=[avg_revenue_inr, revenue_inr],
        labels={"x": "", "y": "Revenue (INR)"},
        title="Average vs Predicted Revenue"
    )
    st.plotly_chart(fig)

    # Bar Graph for Instant Analysis
    st.subheader("Quick Footfall Analysis")
    avg_footfall = df["actual_footfall"].mean()
    fig = px.bar(x=["Average Footfall", "Predicted Footfall"], y=[avg_footfall, predicted_footfall],
                 labels={"x": "", "y": "Footfall"}, title="Average vs Predicted Footfall")
    st.plotly_chart(fig)

    # Future Footfall Predictions (2024-2035)
    st.subheader("Future Footfall Predictions (2024-2035)")
    future_year = st.slider("Select a future year to predict footfall:", min_value=2025, max_value=2035, step=1, value=2030)

    # Calculate future footfall using a 3.8% annual growth rate
    base_footfall = predicted_footfall  # 2024 predicted footfall
    growth_rate = 0.038  # 3.8% annual growth rate (IATA)
    years_range = range(2024, future_year + 1)
    future_footfalls = [base_footfall * (1 + growth_rate) ** (year - 2024) for year in years_range]

    # Create a DataFrame for visualization
    future_df = pd.DataFrame({
        "Year": list(years_range),
        "Predicted Footfall": future_footfalls
    })

    # Visualize the trend with Plotly
    fig = px.line(future_df, x="Year", y="Predicted Footfall", markers=True,
                  title=f"Footfall Trend from 2024 to {future_year}")
    st.plotly_chart(fig)

    # Display the predicted footfall for the selected year
    future_predicted_footfall = future_footfalls[-1]
    st.write(f"### Expected Footfall in {future_year}: {future_predicted_footfall:,.0f}")

    # Sensitivity Analysis
    st.subheader("Sensitivity Analysis")
    st.write("How does footfall change with different parameters?")
    
    # Test different temperatures
    temp_range = [15, 25, 35]
    temp_predictions = []
    for temp in temp_range:
        temp_input = input_data.copy()
        temp_input["temperature"] = temp
        temp_pred = model.predict(temp_input)[0]
        if selected_weekday == "Weekend":
            temp_pred *= 1.15
        temp_pred *= seasonality_multipliers.get(selected_season, 1.0)
        temp_predictions.append(temp_pred)
    
    # Test weekday vs weekend
    weekday_input = input_data.copy()
    weekday_input["weekday_weekend"] = 0
    weekday_pred = model.predict(weekday_input)[0] * seasonality_multipliers.get(selected_season, 1.0)
    
    weekend_input = input_data.copy()
    weekend_input["weekday_weekend"] = 1
    weekend_pred = model.predict(weekend_input)[0] * 1.15 * seasonality_multipliers.get(selected_season, 1.0)

    # Display sensitivity results
    st.write("#### Footfall by Temperature (°C):")
    temp_df = pd.DataFrame({"Temperature (°C)": temp_range, "Predicted Footfall": temp_predictions})
    st.write(temp_df)
    
    st.write("#### Footfall by Day Type:")
    day_df = pd.DataFrame({"Day Type": ["Weekday", "Weekend"], "Predicted Footfall": [weekday_pred, weekend_pred]})
    st.write(day_df)

    # Export Predictions
    st.subheader("Export Predictions")
    export_data = pd.DataFrame({
        "Predicted Footfall": [predicted_footfall],
        "Confidence Interval (plus or minus)": [confidence_interval],
        "Estimated Revenue (INR)": [revenue_inr],
        "Future Year": [future_year],
        "Future Predicted Footfall": [future_predicted_footfall]
    })
    
    # CSV Download
    csv_buffer = StringIO()  # Updated to use StringIO directly
    export_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"footfall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # PDF Download
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    # Add title
    c.drawString(100, 750, "Airport Footfall Prediction Report")

    # Add prediction data
    y_position = 700
    c.drawString(100, y_position, f"Predicted Footfall: {predicted_footfall:,.0f}")
    y_position -= 20
    c.drawString(100, y_position, f"Confidence Interval (plus or minus): {confidence_interval:,.0f}")
    y_position -= 20
    c.drawString(100, y_position, f"Estimated Revenue (INR): {revenue_inr:,.2f}")
    y_position -= 20
    c.drawString(100, y_position, f"Future Year: {future_year}")
    y_position -= 20
    c.drawString(100, y_position, f"Future Predicted Footfall: {future_predicted_footfall:,.0f}")

    # Finalize the PDF
    c.showPage()
    c.save()

    # Move the buffer position to the beginning
    pdf_buffer.seek(0)

    # Add PDF download button
    st.download_button(
        label="Download Predictions as PDF",
        data=pdf_buffer,
        file_name=f"footfall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Footer
st.write("---")
st.write("Built with ❤ by Airport Footfall Predictor Team")

# Feedback Form Section
st.write("---")
st.subheader("Feedback Form")
st.write("We value your feedback! Please let us know about your experience.")

# Define the questions
questions = [
    "Were you satisfied with our platform?",
    "How much satisfied are you with the output?",
    "Did our services satisfy your needs?",
    "How quickly do you think you got your answer once the outputs were produced?",
    "Did you like the feature of downloading using CSV or PDF, which saves time as well?"
]

# Create sliders for each question
feedback_scores = {}
for question in questions:
    score = st.slider(
        label=question,
        min_value=1,
        max_value=10,
        value=5,  # Default to neutral
        step=1,
        format="%d",
        help="1 = Dissatisfied, 5 = Neutral, 10 = Totally Satisfied"
    )
    feedback_scores[question] = score

# Add a submit button
if st.button("Submit Feedback"):
    st.write("Thank you for your feedback!")
    for question, score in feedback_scores.items():
        st.write(f"{question}: {score}/10")

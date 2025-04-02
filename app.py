import streamlit as st
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 import plotly.express as px
 from sklearn.model_selection import train_test_split
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 from io import BytesIO, StringIO
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
 st.title("Syncro - Airport Footfall Prediction and Analysis")
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
 
     # Make predictions on the test set
     y_pred = model.predict(X_test)
 
     # Calculate R-squared (R²) score
     r2 = r2_score(y_test, y_pred)
 
     # Display the result in Streamlit
     st.subheader("Model Performance Metrics")
     st.write(f"### R² Score: {r2:.4f}")
 
     # Interpretation
     if r2 >= 0.75:
         st.success("Great! The model explains a high proportion of variance.")
     elif r2 >= 0.50:
         st.warning("The model performs moderately well but could be improved.")
     else:
         st.error("The model has low explanatory power. Consider feature tuning.")
 
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
     adjusted_fare = weighted_fare * (1 + (fare_multipliers[selected_passenger_class] - 1) * 0.3)
 
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
     avg_revenue_usd = avg_footfall * adjusted_fare
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
     base_footfall = predicted_footfall
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
     csv_buffer = StringIO()
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
     width, height = letter  # Page dimensions (612 x 792 points)
 
     # Helper function to draw wrapped text
     def draw_wrapped_text(canvas, text, x, y, max_width, line_height):
         words = text.split()
         lines = []
         current_line = []
         current_width = 0
 
         for word in words:
             word_width = canvas.stringWidth(word + " ", "Helvetica", 12)
             if current_width + word_width <= max_width:
                 current_line.append(word)
                 current_width += word_width
             else:
                 lines.append(" ".join(current_line))
                 current_line = [word]
                 current_width = word_width
         if current_line:
 # ... (Previous code remains unchanged until the export section) ...
 
 # Export Predictions
 st.subheader("Export Predictions")
 export_data = pd.DataFrame({
     "Predicted Footfall": [predicted_footfall],
     "Confidence Interval (plus or minus)": [confidence_interval],
     "Estimated Revenue (INR)": [revenue_inr],
     "Future Year": [future_year],
     "Future Predicted Footfall": [future_predicted_footfall]
 })
 
 # Add data for comparison graphs to export
 comparison_data = pd.DataFrame({
     "Category": ["Average Footfall", "Predicted Footfall", "Average Revenue (INR)", "Predicted Revenue (INR)"],
     "Value": [avg_footfall, predicted_footfall, avg_revenue_inr, revenue_inr]
 })
 
 # Combine export data and comparison data
 export_data_full = pd.concat([export_data, comparison_data], axis=0, ignore_index=True)
 
 # CSV Download
 csv_buffer = StringIO()
 export_data_full.to_csv(csv_buffer, index=False)
 st.download_button(
     label="Download Predictions as CSV",
     data=csv_buffer.getvalue(),
     file_name=f"footfall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
     mime="text/csv"
 )
 
 # PDF Download with Visualizations
 pdf_buffer = BytesIO()
 c = canvas.Canvas(pdf_buffer, pagesize=letter)
 width, height = letter  # Page dimensions (612 x 792 points)
 
 # Helper function to draw wrapped text
 def draw_wrapped_text(canvas, text, x, y, max_width, line_height):
     words = text.split()
     lines = []
     current_line = []
     current_width = 0
 
     for word in words:
         word_width = canvas.stringWidth(word + " ", "Helvetica", 12)
         if current_width + word_width <= max_width:
             current_line.append(word)
             current_width += word_width
         else:
             lines.append(" ".join(current_line))
 
         for line in lines:
             canvas.drawString(x, y, line)
             y -= line_height
         return y
 
     # Set font
     c.setFont("Helvetica-Bold", 16)
 
     # Page 1: Title and Introduction
     c.drawString(100, 750, "Syncro - Airport Footfall Prediction Report")
     c.setFont("Helvetica", 12)
     y_position = 720
 
     # Introduction
     intro_text = (
         f"This report provides a detailed analysis of predicted airport footfall for {selected_airport} "
         f"based on the following inputs: Season: {selected_season}, Flight Type: {selected_flight_type}, "
         f"Year: {selected_year}, Day: {selected_weekday}, Temperature: {selected_temperature}°C, "
         f"Weather: {selected_weather}, Peak Season: {selected_peak_season}, Holiday: {selected_holiday}, "
         f"Passenger Class: {selected_passenger_class}. The predictions are generated using a RandomForestRegressor model."
     )
     y_position = draw_wrapped_text(c, intro_text, 100, y_position, 400, 15)
 
     # Section: Prediction Summary
     y_position -= 20
     c.setFont("Helvetica-Bold", 14)
     c.drawString(100, y_position, "Prediction Summary")
     y_position -= 20
     c.setFont("Helvetica", 12)
 
     prediction_text = (
         f"Predicted Footfall: {predicted_footfall:,.0f} passengers (Confidence Interval: ±{confidence_interval:,.0f})\n"
         f"Estimated Daily Revenue: {'₹' + f'{revenue_inr:,.2f}' if revenue_inr <= 10000000 else f'{revenue_inr/10000000:,.2f} Crores'}\n"
         f"Future Predicted Footfall (Year {future_year}): {future_predicted_footfall:,.0f} passengers"
     )
     for line in prediction_text.split("\n"):
         c.drawString(100, y_position, line)
         y_position -= 15
 
     # Section: Model Performance
     y_position -= 20
     c.setFont("Helvetica-Bold", 14)
     c.drawString(100, y_position, "Model Performance")
     y_position -= 20
     c.setFont("Helvetica", 12)
 
     model_performance_text = (
         f"R² Score: {r2:.4f}\n"
         f"Interpretation: {('Great! The model explains a high proportion of variance.' if r2 >= 0.75 else 'The model performs moderately well but could be improved.' if r2 >= 0.50 else 'The model has low explanatory power. Consider feature tuning.')}"
     )
     for line in model_performance_text.split("\n"):
         c.drawString(100, y_position, line)
         y_position -= 15
 
     # Check if we need a new page
     if y_position < 100:
         c.showPage()
         y_position = 750
         c.setFont("Helvetica", 12)
 
     # Section: Contextual Analysis
     y_position -= 20
     c.setFont("Helvetica-Bold", 14)
     c.drawString(100, y_position, "Contextual Analysis")
     y_position -= 20
     c.setFont("Helvetica", 12)
 
     # Analyze user inputs and provide insights
     avg_airport_footfall = df[df["airport"] == selected_airport]["actual_footfall"].mean()
     context_analysis = (
         f"For {selected_airport}, the predicted footfall of {predicted_footfall:,.0f} passengers is "
         f"{'above' if predicted_footfall > avg_airport_footfall else 'below'} the historical average of {avg_airport_footfall:,.0f} passengers. "
         f"Given that you selected a {selected_weekday.lower()} during {selected_season} with {selected_weather.lower()} weather, "
         f"{'and considering it is a peak season' if selected_peak_season == 'Yes' else ''} "
         f"{'and a holiday period' if selected_holiday == 'Yes' else ''}, "
         f"the demand for airport services is expected to be "
         f"{'significantly high' if (selected_weekday == 'Weekend' or selected_peak_season == 'Yes' or selected_holiday == 'Yes') else 'moderate'}. "
         f"The temperature of {selected_temperature}°C'sensitive to temperature, with optimal conditions around 25°C. Weekends show a significant increase, suggesting higher demand for leisure travel."
     )
     y_position = draw_wrapped_text(c, context_analysis, 100, y_position, 400, 15)
 
     # Check if we need a new page
     if y_position < 100:
         c.showPage()
         y_position = 750
         c.setFont("Helvetica", 12)
 
     # Section: Operational Recommendations
     y_position -= 20
     c.setFont("Helvetica-Bold", 14)
     c.drawString(100, y_position, "Operational Recommendations")
     y_position -= 20
     c.setFont("Helvetica", 12)
 
     recommendations = (
         f"1. **Staffing and Logistics**: With a predicted footfall of {predicted_footfall:,.0f}, ensure adequate staffing for check-in counters, security, and customer service, especially since it’s a {selected_weekday.lower()} "
         f"{'during peak season' if selected_peak_season == 'Yes' else ''}. Consider increasing ground staff by 20-30% if footfall exceeds historical averages.\n"
         f"2. **Resource Allocation**: The {selected_flight_type.lower()} flight focus suggests prioritizing {'domestic terminal operations' if selected_flight_type == 'Domestic' else 'international terminal operations, including customs and immigration services'}. "
         f"For {selected_passenger_class.lower()} passengers, {'ensure availability of budget amenities' if selected_passenger_class == 'Economy' else 'enhance premium services like lounges and fast-track security'}.\n"
         f"3. **Weather Preparedness**: With {selected_weather.lower()} weather and a temperature of {selected_temperature}°C, "
         f"{'prepare for potential delays by coordinating with airlines' if selected_weather == 'Bad' else 'leverage the pleasant conditions to enhance passenger experience with outdoor amenities'}.\n"
         f"4. **Marketing Opportunities**: Given the {'high' if (selected_weekday == 'Weekend' or selected_peak_season == 'Yes' or selected_holiday == 'Yes') else 'moderate'} demand, consider targeted promotions for {selected_passenger_class.lower()} travelers to maximize revenue."
     )
     y_position = draw_wrapped_text(c, recommendations, 100, y_position, 400, 15)
 
     # Check if we need a new page
     if y_position < 100:
         c.showPage()
         y_position = 750
         c.setFont("Helvetica", 12)
 
     # Section: Sensitivity Analysis
     y_position -= 20
     c.setFont("Helvetica-Bold", 14)
     c.drawString(100, y_position, "Contextual Analysis")
     y_position -= 20
     c.setFont("Helvetica", 12)
 
     # Analyze user inputs and provide insights
     avg_airport_footfall = df[df["airport"] == selected_airport]["actual_footfall"].mean()
     context_analysis = (
         f"For {selected_airport}, the predicted footfall of {predicted_footfall:,.0f} passengers is "
         f"{'above' if predicted_footfall > avg_airport_footfall else 'below'} the historical average of {avg_airport_footfall:,.0f} passengers. "
         f"Given that you selected a {selected_weekday.lower()} during {selected_season} with {selected_weather.lower()} weather, "
         f"{'and considering it is a peak season' if selected_peak_season == 'Yes' else ''} "
         f"{'and a holiday period' if selected_holiday == 'Yes' else ''}, "
         f"the demand for airport services is expected to be "
         f"{'significantly high' if (selected_weekday == 'Weekend' or selected_peak_season == 'Yes' or selected_holiday == 'Yes') else 'moderate'}. "
         f"The temperature of {selected_temperature}°C may {'increase passenger comfort, potentially boosting footfall' if 15 <= selected_temperature <= 25 else 'deter passengers due to extreme weather'}. "
         f"The choice of {selected_flight_type.lower()} flights and {selected_passenger_class.lower()} class suggests a focus on "
         f"{'cost-conscious travelers' if selected_passenger_class == 'Economy' else 'premium travelers willing to pay for enhanced services'}."
     )
     y_position = draw_wrapped_text(c, context_analysis, 100, y_position, 400, 15)
 
     # Check if we need a new page
     if y_position < 100:
         c.showPage()
         y_position = 750
         c.setFont("Helvetica", 12)
 
     # Section: Conclusion
     y_position -= 20
     c.setFont("Helvetica-Bold", 14)
     c.drawString(100, y_position, "Conclusion")
     y_position -= 20
     c.setFont("Helvetica", 12)
 
     conclusion_text = (
         f"This report highlights a predicted footfall of {predicted_footfall:,.0f} passengers for {selected_airport} under the specified conditions. "
         f"The insights provided can help optimize airport operations, improve passenger experience, and maximize revenue. "
         f"For further analysis or custom predictions, please contact the Syncro team."
     )
     y_position = draw_wrapped_text(c, conclusion_text, 100, y_position, 400, 15)
 
     # Finalize the PDF
             current_line = [word]
             current_width = word_width
     if current_line:
         lines.append(" ".join(current_line))
 
     for line in lines:
         canvas.drawString(x, y, line)
         y -= line_height
     return y
 
 # Generate and save comparison graphs as images for PDF
 # Revenue Comparison Graph
 revenue_fig = px.bar(
     x=["Average Revenue", "Predicted Revenue"],
     y=[avg_revenue_inr, revenue_inr],
     labels={"x": "", "y": "Revenue (INR)"},
     title="Average vs Predicted Revenue"
 )
 revenue_img_buffer = BytesIO()
 revenue_fig.write_image(revenue_img_buffer, format="png")
 revenue_img_buffer.seek(0)
 
 # Footfall Comparison Graph
 footfall_fig = px.bar(
     x=["Average Footfall", "Predicted Footfall"],
     y=[avg_footfall, predicted_footfall],
     labels={"x": "", "y": "Footfall"},
     title="Average vs Predicted Footfall"
 )
 footfall_img_buffer = BytesIO()
 footfall_fig.write_image(footfall_img_buffer, format="png")
 footfall_img_buffer.seek(0)
 
 # Set font for PDF
 c.setFont("Helvetica-Bold", 16)
 
 # Page 1: Title and Introduction
 c.drawString(100, 750, "Syncro - Airport Footfall Prediction Report")
 c.setFont("Helvetica", 12)
 y_position = 720
 
 intro_text = (
     f"This report provides a detailed analysis of predicted airport footfall for {selected_airport} "
     f"based on the following inputs: Season: {selected_season}, Flight Type: {selected_flight_type}, "
     f"Year: {selected_year}, Day: {selected_weekday}, Temperature: {selected_temperature}°C, "
     f"Weather: {selected_weather}, Peak Season: {selected_peak_season}, Holiday: {selected_holiday}, "
     f"Passenger Class: {selected_passenger_class}. The predictions are generated using a RandomForestRegressor model."
 )
 y_position = draw_wrapped_text(c, intro_text, 100, y_position, 400, 15)
 
 # Section: Prediction Summary
 y_position -= 20
 c.setFont("Helvetica-Bold", 14)
 c.drawString(100, y_position, "Prediction Summary")
 y_position -= 20
 c.setFont("Helvetica", 12)
 
 prediction_text = (
     f"Predicted Footfall: {predicted_footfall:,.0f} passengers (Confidence Interval: ±{confidence_interval:,.0f})\n"
     f"Estimated Daily Revenue: {'₹' + f'{revenue_inr:,.2f}' if revenue_inr <= 10000000 else f'{revenue_inr/10000000:,.2f} Crores'}\n"
     f"Future Predicted Footfall (Year {future_year}): {future_predicted_footfall:,.0f} passengers"
 )
 for line in prediction_text.split("\n"):
     c.drawString(100, y_position, line)
     y_position -= 15
 
 # Section: Visualizations
 y_position -= 20
 c.setFont("Helvetica-Bold", 14)
 c.drawString(100, y_position, "Comparison Visualizations")
 y_position -= 20
 
 # Add Revenue Comparison Graph to PDF
 if y_position < 300:  # Check if we need a new page
     c.showPage()
     c.save()
 
     # Move the buffer position to the beginning
     pdf_buffer.seek(0)
 
     # Add PDF download button
     st.download_button(
         label="Download Detailed Report as PDF",
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
 st.write("Built with ❤️ by Airport Footfall Predictor Team")
 
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
     y_position = 750
 c.setFont("Helvetica", 12)
 c.drawString(100, y_position, "Revenue Comparison")
 y_position -= 20
 c.drawImage(revenue_img_buffer, 100, y_position - 200, width=400, height=200)
 y_position -= 220
 
 # Add Footfall Comparison Graph to PDF
 if y_position < 300:  # Check if we need a new page
     c.showPage()
     y_position = 750
 c.drawString(100, y_position, "Footfall Comparison")
 y_position -= 20
 c.drawImage(footfall_img_buffer, 100, y_position - 200, width=400, height=200)
 y_position -= 220
 
 # Section: Model Performance
 if y_position < 100:
     c.showPage()
     y_position = 750
 c.setFont("Helvetica-Bold", 14)
 c.drawString(100, y_position, "Model Performance")
 y_position -= 20
 c.setFont("Helvetica", 12)
 
 model_performance_text = (
     f"R² Score: {r2:.4f}\n"
     f"Interpretation: {('Great! The model explains a high proportion of variance.' if r2 >= 0.75 else 'The model performs moderately well but could be improved.' if r2 >= 0.50 else 'The model has low explanatory power. Consider feature tuning.')}"
 )
 for line in model_performance_text.split("\n"):
     c.drawString(100, y_position, line)
     y_position -= 15
 
 # ... (Rest of the PDF sections like Contextual Analysis, Recommendations, etc., remain unchanged) ...
 
 # Finalize the PDF
 c.showPage()
 c.save()
 pdf_buffer.seek(0)
 
 # Add PDF download button
 st.download_button(
     label="Download Detailed Report as PDF",
     data=pdf_buffer,
     file_name=f"footfall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
     mime="application/pdf"

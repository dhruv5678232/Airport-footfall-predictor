import pandas as pd
import streamlit as st
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.express as px
from datetime import datetime

# Export Predictions
st.subheader("Export Predictions")

# Sample variables (replace with actual values from your model/context)
predicted_footfall = 50000  # Example value
confidence_interval = 2500  # Example value
revenue_inr = 7500000  # Example value
future_year = 2026  # Example value
future_predicted_footfall = 55000  # Example value
avg_footfall = 45000  # Example value
avg_revenue_inr = 6800000  # Example value
selected_airport = "JFK"  # Example value
selected_season = "Summer"  # Example value
selected_flight_type = "International"  # Example value
selected_year = 2025  # Example value
selected_weekday = "Friday"  # Example value
selected_temperature = 25  # Example value
selected_weather = "Clear"  # Example value
selected_peak_season = "Yes"  # Example value
selected_holiday = "No"  # Example value
selected_passenger_class = "Economy"  # Example value
r2 = 0.85  # Example value

# Export DataFrame
export_data = pd.DataFrame({
    "Predicted Footfall": [predicted_footfall],
    "Confidence Interval (plus or minus)": [confidence_interval],
    "Estimated Revenue (INR)": [revenue_inr],
    "Future Year": [future_year],
    "Future Predicted Footfall": [future_predicted_footfall]
})

# Comparison DataFrame
comparison_data = pd.DataFrame({
    "Category": ["Average Footfall", "Predicted Footfall", "Average Revenue (INR)", "Predicted Revenue (INR)"],
    "Value": [avg_footfall, predicted_footfall, avg_revenue_inr, revenue_inr]
})

# Combine data for export
export_data_full = pd.concat([export_data, comparison_data], ignore_index=True)

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
width, height = letter

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
        lines.append(" ".join(current_line))

    for i, line in enumerate(lines):
        canvas.drawString(x, y - i * line_height, line)
    return y - len(lines) * line_height

# Generate comparison graphs
# Revenue Comparison Graph
revenue_fig = px.bar(
    x=["Average Revenue", "Predicted Revenue"],
    y=[avg_revenue_inr, revenue_inr],
    labels={"x": "", "y": "Revenue (INR)"},
    title="Average vs Predicted Revenue"
)
revenue_img_buffer = BytesIO()
revenue_fig.write_image(revenue_img_buffer, format="png", engine="kaleido")
revenue_img_buffer.seek(0)

# Footfall Comparison Graph
footfall_fig = px.bar(
    x=["Average Footfall", "Predicted Footfall"],
    y=[avg_footfall, predicted_footfall],
    labels={"x": "", "y": "Footfall"},
    title="Average vs Predicted Footfall"
)
footfall_img_buffer = BytesIO()
footfall_fig.write_image(footfall_img_buffer, format="png", engine="kaleido")
footfall_img_buffer.seek(0)

# PDF Content
c.setFont("Helvetica-Bold", 16)
c.drawString(100, 750, "Syncro - Airport Footfall Prediction Report")
c.setFont("Helvetica", 12)
y_position = 720

# Introduction
intro_text = (
    f"This report provides a detailed analysis of predicted airport footfall for {selected_airport} "
    f"based on: Season: {selected_season}, Flight Type: {selected_flight_type}, Year: {selected_year}, "
    f"Day: {selected_weekday}, Temperature: {selected_temperature}°C, Weather: {selected_weather}, "
    f"Peak Season: {selected_peak_season}, Holiday: {selected_holiday}, Passenger Class: {selected_passenger_class}. "
    "Predictions use a RandomForestRegressor model."
)
y_position = draw_wrapped_text(c, intro_text, 100, y_position, 400, 15)

# Prediction Summary
y_position -= 20
c.setFont("Helvetica-Bold", 14)
c.drawString(100, y_position, "Prediction Summary")
y_position -= 20
c.setFont("Helvetica", 12)

prediction_text = (
    f"Predicted Footfall: {predicted_footfall:,.0f} passengers (±{confidence_interval:,.0f})\n"
    f"Estimated Daily Revenue: {'₹' + f'{revenue_inr:,.2f}' if revenue_inr <= 10000000 else f'{revenue_inr/10000000:,.2f} Crores'}\n"
    f"Future Predicted Footfall (Year {future_year}): {future_predicted_footfall:,.0f} passengers"
)
y_position = draw_wrapped_text(c, prediction_text, 100, y_position, 400, 15)

# Visualizations
y_position -= 20
c.setFont("Helvetica-Bold", 14)
c.drawString(100, y_position, "Comparison Visualizations")
y_position -= 20

# Revenue Graph
if y_position < 300:
    c.showPage()
    y_position = 750
c.setFont("Helvetica", 12)
c.drawString(100, y_position, "Revenue Comparison")
y_position -= 20
c.drawImage(ImageReader(revenue_img_buffer), 100, y_position - 200, width=400, height=200)
y_position -= 220

# Footfall Graph
if y_position < 300:
    c.showPage()
    y_position = 750
c.drawString(100, y_position, "Footfall Comparison")
y_position -= 20
c.drawImage(ImageReader(footfall_img_buffer), 100, y_position - 200, width=400, height=200)
y_position -= 220

# Model Performance
if y_position < 100:
    c.showPage()
    y_position = 750
c.setFont("Helvetica-Bold", 14)
c.drawString(100, y_position, "Model Performance")
y_position -= 20
c.setFont("Helvetica", 12)

model_text = (
    f"R² Score: {r2:.4f}\n"
    f"Interpretation: {('Great! High variance explained.' if r2 >= 0.75 else 'Moderate performance.' if r2 >= 0.50 else 'Low explanatory power.')}"
)
y_position = draw_wrapped_text(c, model_text, 100, y_position, 400, 15)

# Finalize PDF
c.showPage()
c.save()
pdf_buffer.seek(0)

# PDF Download Button
st.download_button(
    label="Download Detailed Report as PDF",
    data=pdf_buffer,
    file_name=f"footfall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    mime="application/pdf"
)

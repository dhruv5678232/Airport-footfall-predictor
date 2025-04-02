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
)

# ... (Rest of the code like Footer and Feedback Form remains unchanged) ...

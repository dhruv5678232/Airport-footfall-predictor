# ✅ Check dataset columns
expected_columns = {
    "Airport": "airport",
    "Season": "season",
    "Flight_Type": "flight type",
    "Year": "year",
    "Weekday_Weekend": "weekday/weekend",
    "Load_Factor": "load_factor",
    "Actual_Footfall": "actual_footfall"
}

# ✅ Rename columns if necessary
df = df.rename(columns={k: v for k, v in expected_columns.items() if k in df.columns})

# ✅ Check if all required columns exist
if all(col in df.columns for col in expected_columns.values()):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    # ✅ Select Relevant Features
    features = ["airport", "season", "flight type", "year", "weekday/weekend", "load_factor"]
    target = "actual_footfall"

    # Train-test split (80-20)
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
    st.sidebar.write(f"**MAE:** {mae:.2f}")
    st.sidebar.write(f"**RMSE:** {rmse:.2f}")
    st.sidebar.write(f"**R² Score:** {r2:.2f}")

    st.sidebar.success("Model Trained Successfully ✅")
else:
    st.sidebar.error("Missing columns required for model training.")
    st.write("### Debug: Available Columns in Dataset")
    st.write(df.columns.tolist())  # Print available columns to debug

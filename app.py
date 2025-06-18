import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load models and scaler
scaler = joblib.load("scaler.pkl")
dnn_model = tf.keras.models.load_model("dnn_model.h5")
rf_model = joblib.load("random_forest_model.pkl")

# Load training R2 scores
dnn_r2_score = 0.9769871369988592   #actual DNN R² score
rf_r2_score = 0.9770531030688017   #actual RF R² score

# Streamlit setup
st.set_page_config(page_title="Fuel Efficiency Predictor", layout="centered")
st.title("🚗 Fuel Efficiency Predictor")
st.markdown("Predict fuel efficiency using engine data. You can compare results from Random Forest and DNN models.")
st.info("✅ Note: DNN model has been fixed and now provides reliable predictions.")

# Model selection
model_choice = st.radio("Select Model", ["Random Forest", "Deep Neural Network (DNN)"])

# Input sliders
engine_size = st.slider("Engine Size (L)", 0.5, 6.5, 2.0)
cylinders = st.selectbox("Cylinders", [1, 2, 3, 4, 6, 8])
compression_ratio = st.slider("Compression Ratio", 5.0, 18.0, 10.0)
horsepower = st.slider("Horsepower", 10, 400, 100)
torque = st.slider("Torque (Nm)", 10, 500, 150)
rpm = st.slider("RPM", 1000, 9000, 3000)
load = st.slider("Load Percent", 0, 100, 50)
temp = st.slider("Ambient Temp (°C)", -10, 50, 25)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
fuel_encoded = 0 if fuel_type == "Petrol" else 1

# Input processing
input_data = np.array([[engine_size, cylinders, compression_ratio, horsepower,
                        torque, rpm, load, temp, fuel_encoded]])
scaled_input = scaler.transform(input_data)

# Prediction
if model_choice == "Random Forest":
    prediction = rf_model.predict(scaled_input)[0]
    r2 = rf_r2_score
else:
    prediction = dnn_model.predict(scaled_input)[0][0]
    r2 = dnn_r2_score

# Output
st.subheader("🔍 Predicted Fuel Efficiency")
st.success(f"Estimated: {prediction:.2f} km/l")
st.write(f"🧠 Model Used: **{model_choice}**")
st.write(f"📊 R² Score (Accuracy): **{r2:.2f}**")

if model_choice == "Random Forest":
    # Confidence interval (approximate)
    all_preds = [tree.predict(scaled_input)[0] for tree in rf_model.estimators_]
    std_dev = np.std(all_preds)
    st.write(f"🔎 Confidence Interval (±1σ): ±{std_dev:.2f} km/l")

# Feature visualization
st.subheader("📈 Input Feature Overview")
labels = ["Engine Size", "Cylinders", "Compression Ratio", "Horsepower",
          "Torque", "RPM", "Load %", "Temp", "Fuel Type"]
values = [engine_size, cylinders, compression_ratio, horsepower,
          torque, rpm, load, temp, fuel_encoded]

fig, ax = plt.subplots()
ax.barh(labels, values, color="skyblue")
ax.set_xlabel("Value")
ax.set_title("Input Feature Values")
st.pyplot(fig)

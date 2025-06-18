
# ⛽ Fuel Efficiency Prediction

A machine learning project to predict fuel efficiency (in km/l) based on engine parameters and environmental conditions.

This project is part of a hands-on learning initiative combining both traditional ML and deep learning approaches.

---

## 📌 Objective

To develop a predictive model that estimates the fuel efficiency of an engine using performance-related input features.

---

## 🧩 Problem Statement

> Predict the **fuel efficiency (km/l)** of an engine based on variables like engine size, torque, compression ratio, horsepower, and ambient conditions.

---

## 🗃️ Dataset

- File: `fuel_efficiency_dataset.csv`
- Contains 8+ features including:
  - Engine Size
  - Cylinders
  - Compression Ratio
  - Horsepower
  - Torque
  - RPM
  - Load Percentage
  - Ambient Temperature
  - Fuel Type (Petrol / Diesel)

---

## ⚙️ Machine Learning Models

Two models were trained and evaluated:

### 🔹 Random Forest Regressor
- Tree-based ensemble model
- Robust against overfitting
- Shows feature importance
- **Currently the best performing model**

### 🔹 Deep Neural Network (DNN)
- Implemented using TensorFlow
- Currently **not giving accurate results**
- **Note**: _I will try my level best to fix the DNN model in future iterations._

---

## 🧠 Features Used

| Feature            | Description                              |
|--------------------|------------------------------------------|
| Engine Size        | Size of the engine in liters             |
| Cylinders          | Number of cylinders in engine            |
| Compression Ratio  | Ratio of volume in combustion chamber    |
| Horsepower         | Engine power output                      |
| Torque             | Rotational force                         |
| RPM                | Revolutions per minute                   |
| Load %             | Engine load at runtime                   |
| Ambient Temp       | Environmental temperature                |
| Fuel Type          | Petrol or Diesel                         |

---

## 📊 App

A simple Streamlit app is included for model prediction.

Run it using:

```bash
streamlit run app.py
```

---

## 📂 Folder Structure

```
fuel_efficiency_model/
├── app.py
├── random_forest_model.pkl
├── dnn_model.h5
├── scaler.pkl
├── fuel_efficiency_dataset.csv
├── README.md
```

---

## 🧪 Requirements

To run locally:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

---

## 📫 Author

**Vivekanand Ojha**  
📧 Email: vivekanandojha09@gmail.com  
🔗 GitHub: [@vivekOJ1129](https://github.com/vivekOJ1129)  
🔗 Linkedin: [Vivekanand Ojha](https://www.linkedin.com/in/vivekanand-ojha-485462289/)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

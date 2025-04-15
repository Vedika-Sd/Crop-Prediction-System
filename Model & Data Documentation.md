
# 📘 Model & Data Documentation

## 🧠 Machine Learning Model

The core of the system is a **Random Forest Classifier** trained on a trusted agricultural dataset. The model predicts the most suitable crops based on:

- **Input features**:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - pH Level
  - Rainfall
  - Temperature

The model was trained using `scikit-learn` with an 80:20 train-test split. Label encoding was applied to the target crop names. The classifier achieved an accuracy of **over 87%**, demonstrating strong generalization for real-world soil and weather inputs.

---

## 🧹 Data Preprocessing

- All categorical labels (crop names) were encoded using `LabelEncoder`.
- Numerical features were checked for consistency; missing or invalid values were handled during model preparation.
- The dataset used for prediction (`crop_data.csv`) is cleaned and verified, making it **reliable for crop recommendations**.

---

## ⚠️ Market Price Data (Live + Fallback)

- The system fetches **live crop prices** using the [data.gov.in API](https://data.gov.in/). If the API is unavailable, it falls back to a local CSV file (`crop_price.csv`).
- **Important Note**: The fallback dataset contains approximate or placeholder values. These are **not always accurate**, and some prices may be outdated or misaligned due to data collection limitations.
- In rare cases, **both the API and fallback CSV may fail** due to network issues or server-side errors. The UI handles these failures gracefully, but the user should be informed of the potential inaccuracies.

---

## 💸 Profit Calculator Logic

The profit calculator is built with a **basic formula**:

```python
profit = (yield_kg * market_price) - total_cost
profit_per_acre = profit / area
```

This provides a **rough estimation** of net profit. It does **not** consider factors like labor, fertilizer breakdown, irrigation cost, regional subsidies, or seasonal price fluctuations. This component is intended as a **simple guide**, not a financial forecasting tool.

---

## ✅ Summary

- ✅ **Crop prediction** is accurate and data-driven.
- ⚠️ **Price data** is useful but may not be 100% reliable.
- ➕ The app is designed to be informative and interactive but is not a substitute for expert agricultural advice.
  
---

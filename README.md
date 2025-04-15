# 🌾 CropGuru AI – Smart Crop Prediction System

**CropGuru AI** is an intelligent web-based system that helps farmers choose the most suitable crops based on soil and weather conditions. Using a trained machine learning model, the system recommends the top 3 crops tailored to the user's input while also providing current market prices, profit calculations, and AI-powered crop alternatives.

---

## 🚀 Features

- ✅ **Top-3 Crop Recommendations** using a Random Forest model trained on soil and climate features.
- 📊 **Live Market Price Fetching** from the [data.gov.in API](https://data.gov.in) with offline fallback support.
- 💸 **Profit Calculator** to estimate earnings based on yield, cost, and land area.
- 🌿 **Crop Alternatives** using cosine similarity with AI-generated justifications via Gemini.
- 🤖 **Agritech AI Assistant** (Gemini) to answer farming-related questions in natural language.
- 🎨 **Interactive UI** built using **Streamlit** with Lottie animations and clean layout.

---

## 🧠 Machine Learning

- **Model Used**: Random Forest Classifier  
- **Input Features**:  
  - Nitrogen (N)  
  - Phosphorus (P)  
  - Potassium (K)  
  - pH Level  
  - Rainfall  
  - Temperature  
- **Accuracy**: 87%+ on test data  
- **Training**: Performed using `sklearn` with label encoding and a train-test split of 80:20.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python (Pandas, NumPy, Scikit-learn)  
- **Visualization**: Matplotlib  
- **External APIs**:  
  - [Data.gov.in](https://data.gov.in/) for crop prices  
  - Gemini AI (Google Generative AI) for crop justifications & chat

---

## 📂 Project Structure

├── app.py # Main application script 
├── crop_data.csv # Dataset used for training 
├── crop_price.csv # Fallback market price data 
├── crop_pred.ipynb # Notebook used to try various models
├── Animation.json # Lottie animation file 
└── README.md # Project description

---

## 📸 Screenshots
![1Homepage](https://github.com/user-attachments/assets/eb5665a5-7f5e-4c83-9709-84c005ad722d)
![2crop_page](https://github.com/user-attachments/assets/a9fc3e70-09bf-48c8-89a1-0f4daeb16505)
![3calci](https://github.com/user-attachments/assets/6f4e5523-035c-4457-8d87-4dced99e3a0a)
![4alternative](https://github.com/user-attachments/assets/6ece61e5-bd43-44f9-97a5-a0578df6940c)
![5ai](https://github.com/user-attachments/assets/5147301c-f3a2-4b47-8f2b-20e71eb13b94)

---
1. **Install dependencies**  
```bash
pip install streamlit scikit-learn pandas matplotlib google-generativeai
Run the app
```
2. Run the app
   ```bash
   streamlit run app.py
   or
   python -m streamlit run app.py
  '''
3.Enter soil parameters and explore the top crop recommendations, profitability, and more!

---

## 👩‍🌾 Use Case
CropGuru AI empowers farmers, agri-researchers, and agribusinesses to make data-driven crop choices by blending machine learning, market trends, and AI insights into one accessible platform.

---

## 📜 License
This project is open-source under the MIT License.

---

## 💡 Author
Developed by: [Vedika Sanjay Sardeshmukh]
📧 Email: [vedikasardeshmukh7@gmail.com]
🌐 GitHub: github.com/Vedika-Sd

---

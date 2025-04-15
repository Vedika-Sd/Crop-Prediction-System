# Smart crop prediction system
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from streamlit_lottie import st_lottie
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# --- Setup ---
st.set_page_config(page_title="CropGuru AI", page_icon="🌾", layout="wide")

# Load Animation
@st.cache_data
def load_lottie(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
crop_animation = load_lottie("Animation.json")

# Display Header
st_lottie(crop_animation, height=300)
st.title("🌾 CropGuru AI")
st.subheader("Smart Crop Prdiction System")
st.caption("An intelligent platform for farmers to make informed crop decisions using AI and data!")

# --- Load Data ---
data = pd.read_csv("crop_data.csv")
label_encoder = LabelEncoder()
data["CROP"] = label_encoder.fit_transform(data["CROP"])
X = data.drop(columns=["CROP"])
y = data["CROP"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Crop Prices Fetch ---
def get_crop_prices():
    try:
        API_KEY = "579b464db66ec23bdd00000108e19cfc7e9a4b9168c53b7a8beb263a"
        DATASET_ID = "9ef84268-d588-465a-a308-a864a43d0070"
        url = f"https://api.data.gov.in/resource/{DATASET_ID}?api-key={API_KEY}&format=json&limit=100"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data["records"]) if "records" in data else pd.DataFrame()

        if not df.empty:
            df = df[["commodity", "modal_price"]]
            df.columns = ["Crop", "Price"]
        else:
            raise ValueError("Empty API data")
    except:
        df = pd.read_csv("crop_price.csv")
        df.columns = [col.strip().title() for col in df.columns]  
    df["Crop"] = df["Crop"].str.strip().str.upper()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna()

# --- Gemini AI Setup ---
genai.configure(api_key="AIzaSyC0aqfKtOsYNFeHmsEQVMXek1ONb0wK9QM")
model = genai.GenerativeModel("gemini-2.0-flash")

def query_gemini_model(question):
    prompt = "You are an expert Agritech AI Assistant helping Indian farmers with clear and simple answers. Answer this:\n\n"
    response = model.generate_content(prompt + question)
    return response.text

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🌾 Recommend Crops", 
    "💸 Profit Calculator", 
    "🌿 Crop Alternatives", 
    "🤖 Chat with Agritech AI"
])

# --- 🌾 Recommend Crops ---
with tab1:
    st.header("🌾 Crop Recommendation Engine")
    col1, col2, col3 = st.columns(3)
    N = col1.number_input("Nitrogen (N)", min_value=60.0)
    P = col2.number_input("Phosphorus (P)", min_value=40.0)
    K = col3.number_input("Potassium (K)", min_value=30.0)

    col4, col5, col6 = st.columns(3)
    pH = col4.number_input("pH Level", min_value=6)
    rainfall = col5.number_input("Rainfall (mm)", min_value=300.0)
    temperature = col6.number_input("Temperature (°C)", min_value=20.0)

    def show_top_crops(crops, prices_df):
        st.subheader("✅ Top 3 Recommended Crops:")
        crop_names = []
        crop_prices = []

        for i, crop in enumerate(crops, start=1):
            name = crop.title()
            crop_names.append(name)
            st.markdown(f"**{i}. {name}**")

            crop_upper = crop.strip().upper()
            match = prices_df[prices_df["Crop"] == crop_upper]
            if not match.empty:
                price = match["Price"].values[0]
                crop_prices.append(price)
            else:
                crop_prices.append(0)

        # 🌾 Price Bar Chart
        st.subheader("📊 Market Prices of Suggested Crops:")
        fig, ax = plt.subplots()
        ax.bar(crop_names, crop_prices, color='green')
        ax.set_ylabel("Price (₹)")
        ax.set_title("Top Crops vs Market Price")
        st.pyplot(fig)

    if st.button("🔍 Predict Best Crops"):
        user_input = np.array([[N, P, K, pH, rainfall, temperature]])
        crop_probs = rf_model.predict_proba(user_input)[0]
        top_idxs = np.argsort(crop_probs)[-3:][::-1]
        top_crops = label_encoder.inverse_transform(top_idxs)
        prices_df = get_crop_prices()
        show_top_crops(top_crops, prices_df)

# --- 💸 Profit Calculator ---
with tab2:
    st.header("💸 Crop Profitability Calculator")
    area = st.number_input("Farming Area (in acres)", min_value=0.0)
    yield_kg = st.number_input("Expected Yield (kg)", min_value=0.0)
    market_price = st.number_input("Market Price (₹/kg)", min_value=0.0)
    cost = st.number_input("Total Cultivation Cost (₹)", min_value=0.0)

    if st.button("🧮 Calculate Profit"):
        revenue = yield_kg * market_price
        profit = revenue - cost
        profit_per_acre = profit / area if area else 0
        st.success(f"Estimated Net Profit: ₹{profit}")
        st.info(f"Profit per Acre: ₹{profit_per_acre:.2f}")

# --- 🌿 Crop Alternatives ---
with tab3:
    st.header("🌿 Alternative Crop Suggestion")
    crop_input = st.text_input("Enter a Crop Name")

    if st.button("🔁 Find Alternatives"):
        try:
            # Match input with label encoder
            all_crops = label_encoder.classes_
            matched_crop = next((c for c in all_crops if c.upper() == crop_input.strip().upper()), None)

            if matched_crop:
                label = label_encoder.transform([matched_crop])[0]
                crop_row = data[data["CROP"] == label].drop(columns=["CROP"])
                crop_vector = crop_row.values[0].reshape(1, -1)

                sim_scores = cosine_similarity(X, crop_vector).flatten()

                data["similarity"] = sim_scores
                similar = data[data["CROP"] != label].sort_values("similarity", ascending=False)

                alt_labels = similar["CROP"].unique()[:3]
                alt_crops = label_encoder.inverse_transform(alt_labels)

                st.subheader(f"📌 Top 3 Alternatives to {matched_crop.title()}:")
                for alt in alt_crops:
                    st.markdown(f"- {alt.title()}")

                st.subheader("🧠 Why these alternatives?")
                for alt in alt_crops:
                    reason = query_gemini_model(f"In 1-2 lines, explain why {alt.title()} is a good alternative to {matched_crop.title()}.")
                    st.write(f"**{alt.title()}**: {reason}")
            else:
                st.error("Crop not found. Please check spelling.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- 🤖 Chat with Agritech AI ---
with tab4:
    st.header("🤖 Ask Anything to Agritech AI Assistant")
    user_question = st.text_input("Type your question about farming, pricing, or crop care")

    if st.button("💬 Get AI Answer"):
        if user_question.strip():
            response = query_gemini_model(user_question)
            st.markdown("**🧠 AI Response:**")
            st.write(response)
        else:
            st.warning("Please enter a valid question.")

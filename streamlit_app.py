import streamlit as st
import requests

st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

st.title("🏦 Bank Churn Predictor")
st.write("Saisis les features, puis clique **Predict** pour appeler l’API.")

# ✅ URL de ton API Azure
API_URL = "https://churn-api-ines-060126.azurewebsites.net/predict"

with st.form("predict_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=650, step=1)
    balance = st.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
    tenure = st.number_input("Tenure", min_value=0, max_value=50, value=5, step=1)
    products = st.number_input("Products", min_value=0, max_value=10, value=2, step=1)
    is_active = st.selectbox("Is active (0/1)", [0, 1], index=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": int(age),
        "credit_score": int(credit_score),
        "balance": float(balance),
        "tenure": int(tenure),
        "products": int(products),
        "is_active": int(is_active),
    }

    st.write("📦 Payload envoyé à l'API :")
    st.json(payload)

    try:
        r = requests.post(API_URL, json=payload, timeout=60)

        st.write(f"✅ Status code: {r.status_code}")

        # Essaye de lire JSON même si erreur
        try:
            data = r.json()
        except Exception:
            st.error("Réponse non-JSON reçue :")
            st.text(r.text)
            st.stop()

        if r.status_code >= 400:
            st.error("❌ Erreur API :")
            st.json(data)
            st.stop()

        st.success("✅ Réponse API :")
        st.json(data)

        # Bonus: affichage joli si ton API renvoie des champs connus
        if "prediction" in data:
            st.write("🎯 Prediction:", data["prediction"])
        if "probability" in data:
            st.write("📊 Probability:", data["probability"])

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Impossible d'appeler l'API: {e}")
        st.info("Vérifie que /predict marche dans /docs et que l’URL est correcte.")

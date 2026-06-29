import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import predict_suzlon

st.set_page_config(page_title="SUZON AI Predictor", layout="wide")

# Blue gradient background
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='font-size: 80px; color: #00f5ff; text-align: center; font-weight: bold;'>SUZON</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #ffd700; text-align: center;'>Accurately Predict Stock Prices</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>2026 | 2027 | 2028 | 2030</h4>", unsafe_allow_html=True)
st.markdown("<p style='background: red; padding: 8px; border-radius: 8px; text-align: center; font-weight: bold;'>AI analysis trend (free)</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Results within 3 seconds</p>", unsafe_allow_html=True)

# Input + Button
col1, col2 = st.columns([3, 1])
with col1:
    stock = st.text_input("Enter stock name/code", value="SUZLON.NS", label_visibility="collapsed")
with col2:
    btn = st.button("Check now", use_container_width=True, type="primary")

if btn:
    with st.spinner("AI analyzing..."):
        result = predict_suzlon()

    if result:
        st.markdown("---")
        st.metric("Current Price", f"₹{result['current']:.2f}")

        cols = st.columns(4)
        for i, (year, price) in enumerate(result['predictions'].items()):
            cols[i].metric(f"Target {year}", f"₹{price:.2f}")

        st.warning("⚠️ Ye prediction sirf educational hai. Investment risk apna.")
    else:
        st.error("Data nahi mila")
# app.py

import streamlit as st
from response_module import get_supportive_response

st.set_page_config(page_title="Stress Detector Chatbot", layout="centered")
st.title("🧘‍♀️ Stress Detector Chatbot")
st.markdown("Enter your message below and I'll let you know if you're stressed, plus give a supportive response.")

user_input = st.text_area("Your Message", height=150)

if st.button("Analyze"):
    with st.spinner("Analyzing…"):
        response, score = get_supportive_response(user_input)
    st.markdown(f"**Stress Score:** {score:.2f}")
    st.info(response)

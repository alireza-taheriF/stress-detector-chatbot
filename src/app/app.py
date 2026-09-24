# src/app/app.py

import streamlit as st
from streamlit_chat import message
from response_module import get_supportive_response, lora_adapter_available
import pandas as pd
import os

st.set_page_config(page_title="Stress Detector Chatbot", layout="wide")

# Paths
BASE_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(BASE_DIR, "..", "logs", "user_feedback.csv")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, bot) tuples

st.title("🧘‍♀️ Stress Detector Chatbot")

# Full fine-tune stays selected. LoRA is optional and only loads when chosen.
model_label = st.selectbox(
    "Detection model",
    ("Full fine-tune", "LoRA adapter"),
    index=0,
    help="Full fine-tune is the baseline. LoRA loads distilbert-base-uncased plus the saved adapter.",
)
model_key = "lora" if model_label == "LoRA adapter" else "full"
if not lora_adapter_available():
    st.caption(
        "LoRA adapter folder artifacts/lora-distilbert/ is not present. "
        "The full fine-tune remains the active default."
    )

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your Message:", height=100)
    submit = st.form_submit_button("Send")

if submit and user_input:
    if model_key == "lora" and not lora_adapter_available():
        st.error(
            "LoRA adapter was not found at artifacts/lora-distilbert/. "
            "Leave the selector on Full fine-tune to keep the current model."
        )
    else:
        bot_resp, score = get_supportive_response(user_input, model_key=model_key)
        st.session_state.history.append((user_input, bot_resp, score, model_label))

# Display chat history
for user_msg, bot_msg, score, used_model in st.session_state.history:
    message(user_msg, is_user=True)
    message(bot_msg, is_user=False)
    st.caption(f"Stress Score: {score:.2f} ({used_model})")

# Feedback sliders and submit
if st.session_state.history:
    st.markdown("---")
    st.subheader("Submit Feedback")
    empathy = st.slider("Empathy (1–5)", 1, 5, 3)
    usefulness = st.slider("Usefulness (1–5)", 1, 5, 3)
    clarity = st.slider("Clarity (1–5)", 1, 5, 3)
    feedback = st.text_area("Any comments?", "")
    if st.button("Submit Feedback"):
        # log the last exchange
        last = st.session_state.history[-1]
        df = pd.DataFrame([{
            "user_message": last[0],
            "bot_response": last[1],
            "stress_score": last[2],
            "empathy": empathy,
            "usefulness": usefulness,
            "clarity": clarity,
            "comments": feedback
        }])
        if os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)
        st.success("Thank you for your feedback!")



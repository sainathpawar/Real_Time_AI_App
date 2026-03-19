import streamlit as st
import requests
from requests.exceptions import RequestException

# Configuration

API_URL = "http://localhost:8000/chat"
TIMEOUT_SECONDS = 25 # Slightly higher than backend
MAX_INPUT_LENGTH = 2000 # Keep same as backend

# Page Setup

st.set_page_config(
    page_title="Real-Time AI Support Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Real-Time AI Support Assistant")
st.caption("Intnent-aware . Retireval Augmented . Saas Ready acrhictecture")

# Session State

if "history" not in st.session_state:
    st.session_state.history = []

if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None

# Input Form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Ask your question",
        max_chars=MAX_INPUT_LENGTH, 
        placeholder="e.g, How do I upgrade my billing plan?")
    submitted = st.form_submit_button("Send")

# Submit Handling

if submitted:
    if not user_input or not user_input.strip():
        st.warning("Please enter a message before submitting.")
    else:
        placeholder = st.empty()
        try:
            with placeholder.container():
                st.info("Sending request to backned...")
            response = requests.post(
                API_URL,
                json={"message": user_input},
                timeout=TIMEOUT_SECONDS
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("response", "No response from backend.")
            request_id = data.get("request_id", "N/A")

            # Store Conversation
            st.session_state.history.append(("You", user_input.strip()))
            st.session_state.history.append(("Assistant", answer.strip()))
            st.session_state.last_request_id = request_id

        except RequestException as e:
            st.error(f"Error communicating with backend: {e}")
        finally:
            placeholder.empty()

# Chat History Render
if st.session_state.history:
    st.divider()
    st.subheader("Converstion")

    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

# Footer Info

if st.session_state.last_request_id:
    st.divider()
    st.caption(f"Request ID: {st.session_state.last_request_id}")



import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Medical AI Assistant", layout="centered")

st.title("🩺 Medical AI Assistant")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Message", placeholder="Ask a medical question...")

if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        try:
            response = requests.post(
                API_URL,
                params={"query": user_input},   # matches your FastAPI signature
                timeout=30
            )

            response.raise_for_status()
            data = response.json()

            answer = data.get("answer", "No answer returned.")
            latency = data.get("latency", None)
            sources = data.get("sources", [])

            # Store messages
            st.session_state.chat.append(("user", user_input))
            st.session_state.chat.append(("bot", answer))

            # Optional metadata display
            with st.expander("🔍 Response Details"):
                if latency is not None:
                    st.write(f"⏱ Latency: `{latency:.2f}` seconds")

                if sources:
                    st.subheader("📄 Sources")
                    for i, src in enumerate(sources, 1):
                        st.write(f"**{i}.** {src}")

        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")

# Render chat history
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"🟢 **You:** {msg}")
    else:
        st.markdown(f"⚪ **Bot:** {msg}")
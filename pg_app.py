import streamlit as st
import requests

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="GovPrep AI",
    page_icon="🇮🇳",
    layout="centered"
)

# CHANGED: Point to  local FastAPI server
API_URL = "http://127.0.0.1:8000/chat"

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("GovPrep AI")
    st.write("AI study assistant for preparation.")
    st.write("**Sources:** NCERT Polity, History, Geography")

    st.divider()

    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Main UI
# -----------------------------
st.title("GovPrep AI ")
st.write("Ask questions from NCERT Polity, History and Geography.")

# -----------------------------
# Draw Previous Messages
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # CHANGED: Updated parsing for the new simple string list format
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources Used"):
                for s in msg["sources"]:
                    st.write(f"**Source:** {s}")

# -----------------------------
# Example Questions
# -----------------------------
prompt = None

if len(st.session_state.messages) == 0:
    st.write("💡 Try asking:")
    
    col1, col2, col3 = st.columns(3)
    if col1.button("What is Article 21?"):
        prompt = "What is Article 21?"
    if col2.button("Causes of 1857 Revolt?"):
        prompt = "What were the causes of the Revolt of 1857?"
    if col3.button("Layers of Atmosphere?"):
        prompt = "Explain the layers of the atmosphere."

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Ask your question...")

if user_input:
    prompt = user_input

# -----------------------------
# Process Question
# -----------------------------
if prompt:
    # show user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching NCERT archives..."):
                response = requests.post(
                    API_URL,
                    json={
                        "question": prompt
                    }
                )

                response.raise_for_status()
                result = response.json()

                answer_text = result.get("answer", "I could not find an answer.")
                sources = result.get("sources", [])

                # show answer
                st.write(answer_text)

                # show sources (CHANGED to handle simple strings)
                if sources:
                    with st.expander("📚 Sources Used"):
                        for s in sources:
                            st.write(f"**Source:** {s}")

                # save assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources
                    }
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to GovPrep API. "
                "Make sure FastAPI is running on http://127.0.0.1:8000"
            )
        except requests.exceptions.HTTPError as e:
            st.error(
                f"API Error ({response.status_code}): "
                f"{response.json().get('detail', 'Unknown error')}"
            )
        except Exception:
            st.error("Something went wrong. Please try again.")
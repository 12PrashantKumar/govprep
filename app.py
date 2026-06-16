import streamlit as st
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from memory import ConversationMemory
from generate_v1 import answer

st.set_page_config(page_title="GovPrep AI", page_icon="🇮🇳", layout="centered")

# Initialize session states
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("GovPrep AI")
    st.write("AI study assistant for prep.")
    st.write("**Sources:** NCERT Polity, History, Geography")
    st.divider()
    
    if st.button("Reset Conversation"):
        st.session_state.memory = ConversationMemory()
        st.session_state.messages = []
        st.rerun()

st.title("GovPrep AI: NCERT Doubt Solver")
st.write("Ask about NCERT Polity, History, and Geography.")

# 1. Redraw previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "chunks" in msg and msg["chunks"]:
            with st.expander("📚 Sources used"):
                for c in msg["chunks"]:
                    st.write(f"**Source:** {c['source'].title()} | **Page:** {c['page']} | **Distance:** {c['distance']:.3f}")
                    st.write(c["text"][:150] + "...")
                    st.divider()

# Setup a unified prompt variable
prompt = None


# We only render these if the chat is completely empty!
if len(st.session_state.messages) == 0:
    st.write("💡 **Try asking:**")
    col1, col2, col3 = st.columns(3)
    
    # If a button is clicked, we assign its text to the 'prompt' variable
    if col1.button("What is Article 21?"):
        prompt = "What is Article 21?"
    if col2.button("Causes of 1857 Revolt?"):
        prompt = "What were the main causes of the 1857 revolt?"
    if col3.button("Atmospheric Layers?"):
        prompt = "Explain the layers of the atmosphere."



user_input = st.chat_input("Type your question here...")


if user_input:
    prompt = user_input


# If 'prompt' has text (either from a button click OR the chat box), run the AI!
if prompt:
    # Immediately draw the user's question on screen
    with st.chat_message("user"):
        st.write(prompt)
    
    # Save the user's question to UI memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process the AI response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching NCERT archives..."):
                result = answer(prompt, st.session_state.memory)
                
                # 1. SAFELY EXTRACT DATA: Prevent KeyErrors if the backend changes structure
                response_text = result.get("answer", "I don't have enough information to answer that.")
                source_chunks = result.get("chunks", [])
                
                # 2. Draw the answer
                st.write(response_text)
                
                # 3. CONDITIONAL RENDERING: Only draw the expander IF chunks actually exist
                if source_chunks:
                    with st.expander("📚 Sources used"):
                        for c in source_chunks:
                            st.write(f"**Source:** {c.get('source', 'Unknown').title()} | **Page:** {c.get('page', 'N/A')} | **Distance:** {c.get('distance', 0.0):.3f}")
                            st.write(c.get("text", "")[:150] + "...")
                            st.divider()
                
                # 4. Save the AI's response to UI memory
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "chunks": source_chunks
                })
                
        except Exception as e:
            
            st.error("Oops! The AI servers are currently busy. Please wait a moment and try again.")
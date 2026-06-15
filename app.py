import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent / "scripts"))
from memory import ConversationMemory
from generate_v1 import answer

st.title("GOVPREP")
st.write("Ask about NCERT Polity,history,geography")
mem = ConversationMemory()
question = st.text_input("Your question:")
if question:
    with st.spinner("Thinking"):
        result = answer(question, mem)
    
    st.write(result["answer"])
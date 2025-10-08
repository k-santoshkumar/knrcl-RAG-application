import streamlit as st
from with_embeddings import answer_financial_question
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import re
from PyPDF2 import PdfReader, PdfWriter

def rewrite_followup_question(chat_history, new_question):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(greet in new_question.lower() for greet in greetings):
        return "Hello! How can I assist you today?"

    load_dotenv()
    llm = AzureChatOpenAI(model="gpt-4o", api_version="2025-01-01-preview", temperature=0)
    context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]]
    )

    prompt = f"""
You are an intelligent assistant that rewrites user questions into standalone versions:
--- Chat History ---
{context}
--- New User Question ---
{new_question}

1. Rewrite the user's question into a standalone version (retain meaning and clarity).
2. Understand if the it is a follow up question or not first and then rewrite it accordingly.
3. If it is not a follow up question, rewrite it as a standalone question.
Answer format:
<rewritten standalone question>

Only output the rewritten question. No other explanation.
"""
    rewritten = llm.invoke(prompt).content.strip()
    return rewritten

def main():
    st.set_page_config(
        page_title="Financial Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ca1, ca2 = st.columns([1, 5])
    with ca1:             
        st.image(os.path.abspath(r"images.png"), width=200)
        pass

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Type your message here...")

    # Show the full message history in the UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        # Rewrite follow-up question to standalone and classify
        rewritten_with_type = rewrite_followup_question(st.session_state.messages[-4:], user_input)
        print(f"Rewritten question: {rewritten_with_type}")
        # Append original question to session and show in UI
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Handle the response
        with st.spinner("KNRCL agent is typing..."):
            response = answer_financial_question(rewritten_with_type)  # rewritten_input is used internally

        # Show assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()


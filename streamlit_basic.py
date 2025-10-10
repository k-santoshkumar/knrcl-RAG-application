# import streamlit as st
# from with_embeddings import answer_financial_question
# from langchain_openai import AzureChatOpenAI
# from dotenv import load_dotenv
# import os

# def rewrite_followup_question(chat_history, new_question):
#     greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
#     if any(greet in new_question.lower() for greet in greetings):
#         return "Hello! How can I assist you today?"

#     load_dotenv()
#     llm = AzureChatOpenAI(model="gpt-4o", api_version="2025-01-01-preview", temperature=0)
#     context = "\n".join(
#         [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]]
#     )

#     prompt = f"""
# You are an intelligent assistant that rewrites user questions into standalone versions:
# --- Chat History ---
# {context}
# --- New User Question ---
# {new_question}

# 1. Rewrite the user's question into a standalone version (retain meaning and clarity).
# 2. Understand if the it is a follow up question or not first and then rewrite it accordingly.
# 3. If it is not a follow up question, rewrite it as a standalone question.
# Answer format:
# <rewritten standalone question>

# Only output the rewritten question. No other explanation.
# """
#     rewritten = llm.invoke(prompt).content.strip()
#     return rewritten

# def main():
#     st.set_page_config(
#         page_title="Financial Assistant",
#         page_icon="📊",
#         layout="wide",
#         initial_sidebar_state="expanded",
#     )
#     ca1, ca2 = st.columns([1, 5])
#     with ca1:             
#         st.image(os.path.abspath(r"images.png"), width=200)
#         pass

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     user_input = st.chat_input("Type your message here...")

#     # Show the full message history in the UI
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     if user_input:
#         # Rewrite follow-up question to standalone and classify
#         rewritten_with_type = rewrite_followup_question(st.session_state.messages[-4:], user_input)
#         print(f"Rewritten question: {rewritten_with_type}")
#         # Append original question to session and show in UI
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         # Handle the response
#         with st.spinner("KNRCL agent is typing..."):
#             response = answer_financial_question(rewritten_with_type)  # rewritten_input is used internally

#         # Show assistant response
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         with st.chat_message("assistant"):
#             st.markdown(response)

# if __name__ == "__main__":
#     main()


import streamlit as st
from with_embeddings import answer_financial_question, process_pdf
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
import os
import tempfile
import shutil
import re

# === Load environment ===
load_dotenv()

# === Global Config ===
FAISS_PATH = "faiss_index"
FOLDER_PATH = "./knowledge"

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version="2024-12-01-preview"
)

def rewrite_followup_question(chat_history, new_question):

    new_q = new_question.strip()
    new_q_lower = new_q.lower()

    # Build a compact context (handle empty history)
    try:
        if chat_history:
            context = "\n".join(
                [f"{msg.get('role','user').capitalize()}: {msg.get('content','')}" for msg in chat_history[-6:]]
            )
        else:
            context = "(no prior chat history)"

        prompt = f"""
You are an intelligent assistant that rewrites user questions into standalone versions:
--- Chat History ---
{context}
--- New User Question ---
{new_q}

1. Rewrite the user's question into a standalone version (retain meaning and clarity).
2. Understand if it is a follow up question or not first and then rewrite it accordingly.
3. If it is not a follow up question, rewrite it as a standalone question.

Answer format:
<rewritten standalone question>

Only output the rewritten question. No other explanation.
"""

        llm = AzureChatOpenAI(model="gpt-4o", api_version="2025-01-01-preview", temperature=0)
        resp = llm.invoke(prompt)

        # resp may expose .content (a string) or a more complex structure; handle common cases
        rewritten = None
        if hasattr(resp, "content"):
            rewritten = resp.content
        elif isinstance(resp, dict):
            # try common dict shapes
            rewritten = resp.get("content") or resp.get("text")

        if isinstance(rewritten, list):
            # sometimes content is a list of message objects
            try:
                rewritten = rewritten[0].get("content") if isinstance(rewritten[0], dict) else str(rewritten[0])
            except Exception:
                rewritten = str(rewritten)

        rewritten = (rewritten or "").strip()
        if not rewritten:
            # Fallback to returning the original question if model produced nothing
            return new_q

        return rewritten
    except Exception as e:
        # Log and gracefully fallback to original question
        print("rewrite_followup_question error:", e)
        return new_q

def main():
    st.set_page_config(
        page_title="KNRCL Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 KNRCL Assistant")

    ca1, ca2 = st.columns([1, 5])
    with ca1:
        st.image(os.path.abspath("images.png"), width=200)

    # === Sidebar: PDF Upload ===
    st.sidebar.header("📁 Document Management")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.info(f"{len(uploaded_files)} file(s) selected.")
        if st.sidebar.button("📚 Process & Index PDFs"):
            with st.spinner("Processing and indexing PDFs..."):
                temp_dir = tempfile.mkdtemp()
                all_docs = []
                try:
                    # Read each uploaded file's bytes once, store for processing and saving
                    saved_files = []
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.read()
                        if not file_bytes:
                            # Skip empty uploads
                            print(f"Skipped empty upload: {uploaded_file.name}")
                            continue

                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(file_bytes)

                        docs = process_pdf(temp_path)
                        if docs:
                            all_docs.extend(docs)

                        saved_files.append((uploaded_file.name, file_bytes))

                    if all_docs:
                        if os.path.exists(FAISS_PATH):
                            vs = FAISS.load_local(
                                FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
                            )
                            vs.add_documents(all_docs)
                            vs.save_local(FAISS_PATH)
                        else:
                            vs = FAISS.from_documents(all_docs, embedding_model)
                            vs.save_local(FAISS_PATH)

                        # Save uploaded files permanently using cached bytes
                        os.makedirs(FOLDER_PATH, exist_ok=True)
                        for name, b in saved_files:
                            with open(os.path.join(FOLDER_PATH, name), "wb") as f:
                                f.write(b)

                        st.sidebar.success("✅ PDFs processed and indexed successfully.")
                    else:
                        st.sidebar.error("⚠️ No valid text extracted from uploaded PDFs.")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {e}")
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

    # === Chat Section ===
    if "messages" not in st.session_state:
        # Initialize with a friendly assistant message so the UI isn't empty on first load
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm KNRCL Assistant. Upload PDFs or ask a question to get started."}
        ]

    user_input = st.chat_input("Type your message here...")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:

        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if user_input.lower().strip() in greetings:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("KNRCL agent is thinking..."):
                response = "Hello! How can I assist you today (regarding constructions !!!)?"

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            rewritten_with_type = rewrite_followup_question(st.session_state.messages[-4:], user_input)
            print(f"Rewritten question: {rewritten_with_type}")
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("KNRCL agent is thinking..."):
                response = answer_financial_question(rewritten_with_type)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    main()

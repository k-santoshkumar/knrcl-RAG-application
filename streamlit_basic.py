import streamlit as st
from with_embeddings import answer_financial_question
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import tempfile
import shutil
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

def extract_citations_from_response(response: str) -> Tuple[str, List[Dict]]:
    """
    Extract citations from the response text and return clean text with citations.
    Returns: (clean_text, citations_list)
    """
    # Pattern to find citations like [1], [2,3], etc.
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    
    # Extract all citations
    citations = []
    citation_matches = re.finditer(citation_pattern, response)
    
    for match in citation_matches:
        citation_nums = match.group(1).split(',')
        for num in citation_nums:
            citations.append(int(num.strip()))
    
    # Get unique citation numbers
    unique_citations = sorted(set(citations))
    
    # For now, return the response as-is with placeholder citations
    # You'll need to modify this based on how your with_embeddings module returns citations
    citation_list = []
    for num in unique_citations:
        citation_list.append({
            "number": num,
            "source": f"Document chunk {num}",
            "content": "Citation content would appear here based on actual retrieval"
        })
    
    return response, citation_list

def process_uploaded_files(uploaded_files) -> Dict[str, str]:
    """
    Process uploaded files and save them temporarily.
    Returns dictionary with file names and paths.
    """
    file_paths = {}
    
    if uploaded_files:
        # Create a temporary directory for this session
        if "temp_dir" not in st.session_state:
            st.session_state.temp_dir = tempfile.mkdtemp()
        
        for uploaded_file in uploaded_files:
            # Save file to temporary directory
            file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths[uploaded_file.name] = file_path
            
    return file_paths

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
2. Understand if it is a follow up question or not first and then rewrite it accordingly.
3. If it is not a follow up question, rewrite it as a standalone question.
Answer format:
<rewritten standalone question>
Only output the rewritten question. No other explanation.
"""
    rewritten = llm.invoke(prompt).content.strip()
    return rewritten

def display_citations(citations: List[Dict]):
    """Display citations in an expandable section."""
    if citations:
        with st.expander("📚 Citations & Sources", expanded=False):
            for i, citation in enumerate(citations, 1):
                st.markdown(f"**[{i}]** {citation.get('source', 'Unknown source')}")
                if 'content' in citation:
                    st.caption(citation['content'][:200] + "..." if len(citation['content']) > 200 else citation['content'])
                st.divider()

def display_metadata(metadata: Dict):
    """Display metadata in a structured format."""
    if metadata:
        with st.expander("ℹ️ Response Metadata", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Used", metadata.get("model", "GPT-4"))
                st.metric("Response Time", f"{metadata.get('response_time', 'N/A')} sec")
            
            with col2:
                st.metric("Tokens Used", metadata.get("tokens", "N/A"))
                st.metric("Context Length", metadata.get("context_length", "N/A"))
            
            with col3:
                st.metric("Documents Searched", metadata.get("docs_searched", "N/A"))
                st.metric("Confidence Score", f"{metadata.get('confidence', 'N/A')}%")

def main():
    st.set_page_config(
        page_title="Financial Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Header with logo
    ca1, ca2 = st.columns([1, 5])
    with ca1:             
        if os.path.exists("images.png"):
            st.image("images.png", width=200)
    
    with ca2:
        st.title("📊 KNRCL Financial Assistant")
        st.caption("AI-powered financial document analysis with citation support")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    if "file_metadata" not in st.session_state:
        st.session_state.file_metadata = {}
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files to analyze",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'csv', 'xlsx'],
            help="Upload financial documents for analysis"
        )
        
        if uploaded_files:
            file_paths = process_uploaded_files(uploaded_files)
            st.session_state.uploaded_files = file_paths
            
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Display uploaded files
            st.subheader("Uploaded Files:")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024  # Size in KB
                st.markdown(f"📄 **{file.name}** ({file_size:.1f} KB)")
        
        st.divider()
        
        # Settings section
        st.header("⚙️ Settings")
        
        show_citations = st.checkbox("Show Citations", value=True)
        show_metadata = st.checkbox("Show Metadata", value=True)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Clear files button
        if st.session_state.uploaded_files and st.button("🗑️ Clear Files", type="secondary", use_container_width=True):
            # Clean up temporary directory
            if "temp_dir" in st.session_state and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir)
            st.session_state.uploaded_files = {}
            st.session_state.file_metadata = {}
            st.rerun()
    
    # Main chat interface
    main_container = st.container()
    
    with main_container:
        # Display current context indicator
        if st.session_state.uploaded_files:
            with st.container():
                st.info(f"📎 Analyzing with {len(st.session_state.uploaded_files)} document(s) in context")
        
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Display citations if available and enabled
                if show_citations and "citations" in msg:
                    display_citations(msg["citations"])
                
                # Display metadata if available and enabled
                if show_metadata and "metadata" in msg:
                    display_metadata(msg["metadata"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Rewrite follow-up question to standalone
        rewritten_question = rewrite_followup_question(st.session_state.messages[-4:], user_input)
        print(f"Rewritten question: {rewritten_question}")
        
        # Append original question to session and show in UI
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Handle the response
        with st.spinner("KNRCL agent is processing..."):
            try:
                # Prepare context with uploaded files if available
                context = {
                    "question": rewritten_question,
                    "files": st.session_state.uploaded_files if st.session_state.uploaded_files else None
                }
                
                # Call your existing function
                # You might need to modify this based on your with_embeddings module
                if st.session_state.uploaded_files:
                    # If files are uploaded, pass them to the function
                    # This assumes your answer_financial_question can handle file context
                    response = answer_financial_question(
                        rewritten_question,
                        file_paths=list(st.session_state.uploaded_files.values())
                    )
                else:
                    response = answer_financial_question(rewritten_question)
                
                # Extract citations from response (you'll need to modify this based on your actual implementation)
                clean_response, citations = extract_citations_from_response(response)
                
                # Generate metadata (placeholder - replace with actual metadata from your system)
                metadata = {
                    "model": "GPT-4",
                    "response_time": "2.3",
                    "tokens": "450",
                    "context_length": "2048",
                    "docs_searched": len(st.session_state.uploaded_files) if st.session_state.uploaded_files else "0",
                    "confidence": "92"
                }
                
                # Create message with citations and metadata
                message_data = {
                    "role": "assistant",
                    "content": clean_response,
                    "citations": citations if show_citations else None,
                    "metadata": metadata if show_metadata else None
                }
                
                # Append to messages
                st.session_state.messages.append(message_data)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(clean_response)
                    
                    if show_citations and citations:
                        display_citations(citations)
                    
                    if show_metadata:
                        display_metadata(metadata)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error while processing your request: {str(e)}"
                })

# Cleanup on app termination
def cleanup():
    if "temp_dir" in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

if __name__ == "__main__":
    main()


import os
import traceback
from typing import List, Optional
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity


# === Load environment ===
load_dotenv()

# === Config ===
FOLDER_PATH = r"./knowledge"
FAISS_PATH = "faiss_index"
CHUNK_SIZE = 1000  # characters
# === Models ===
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version="2024-12-01-preview"
)

llm = AzureChatOpenAI(
    model='gpt-4o',
    api_version='2025-01-01-preview',
    temperature=0
)

# === Helper: PDF Processing ===
from typing import List
from langchain.schema import Document
from PyPDF2 import PdfReader
import os

def process_pdf(file_path: str) -> List[Document]:
    """
    Processes a PDF into LangChain Documents with 1-page overlap.
    Each chunk contains text from two consecutive pages 

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List[Document]: A list of LangChain Documents with overlapping page content.
    """
    try:
        reader = PdfReader(file_path)
        file_name = os.path.basename(file_path)
        num_pages = len(reader.pages)
        documents = []

        for i in range(num_pages):
            # Determine the page range for this chunk
            start_page = i
            end_page = min(i + 1, num_pages - 1)

            # Combine the text from current and next page (if exists)
            text = reader.pages[start_page].extract_text() or ""
            if end_page != start_page:  # if there’s a next page
                text += "\n" + (reader.pages[end_page].extract_text() or "")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "filename": file_name,
                            "start_page": start_page + 1,
                            "end_page": end_page + 1,
                        }
                    )
                )

        return documents

    except Exception as e:
        print(f"❌ Error reading PDF {file_path}: {e}")
        return []

# === Helper: TXT Processing ===
def process_txt(file_path: str, chunk_size: int = CHUNK_SIZE) -> List[Document]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        file_name = os.path.basename(file_path)
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return [
            Document(
                page_content=chunk,
                metadata={"filename": file_name, "chunk_id": i + 1}
            )
            for i, chunk in enumerate(chunks) if chunk.strip()
        ]
    except Exception as e:
        print(f"❌ Error reading TXT {file_path}: {e}")
        return []

# === Helper: Extract All Docs ===
def extract_documents_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if os.path.isfile(path):
            if file.endswith(".pdf"):
                docs.extend(process_pdf(path))
            elif file.endswith(".txt"):
                docs.extend(process_txt(path))
    return docs

# === Helper: Load or Create Vectorstore ===
def load_vectorstore() -> Optional[FAISS]:
    try:
        if os.path.exists(FAISS_PATH):
            print("📦 Loading FAISS index...")
            return FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        else:
            print("🆕 No index found.")
            return None
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return None

# === Helper: Reranker ===
def rerank_documents(question: str, documents, embedding_model, top_n: int = 10):
    """
    Re-ranks documents based on cosine similarity to the embedded question.
    
    Args:
        question (str): User's query.
        documents (List[Document]): List of LangChain Document objects.
        embedding_model: Embedding model with .embed_query and .embed_documents methods.
        top_n (int): Number of top documents to return.
    
    Returns:
        List[Document]: Top-N documents with similarity scores in metadata["score"].
    """
    try:
        if not documents:
            raise ValueError("Document list is empty.")
        
        # Embed query and documents
        query_vec = embedding_model.embed_query(question)
        doc_texts = [doc.page_content for doc in documents]
        doc_vecs = embedding_model.embed_documents(doc_texts)
        
        # Compute cosine similarity
        sims = cosine_similarity([query_vec], doc_vecs)[0]
        
        # Pair scores with documents
        scored_docs = []
        for doc, score in zip(documents, sims):
            new_doc = doc
            new_doc.metadata = {**doc.metadata, "score": float(score)}
            scored_docs.append(new_doc)
        
        # Sort by similarity
        ranked = sorted(scored_docs, key=lambda d: d.metadata["score"], reverse=True)
        return ranked[:top_n]
    
    except Exception as e:
        print(f"❌ Error during document re-ranking: {e}")
        return documents[:top_n]  # fallback

# === Prompt Templates ===
subquery_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Given a complex user question, split it into FIVE focused sub-questions that can be used to retrieve relevant information on if it is required.

User question: "{question}"

Sub-questions:
""")

main_prompt_template = """
You are a highly skilled **Financial Analyst Assistant**, trained to extract insights strictly from financial documents.

Your role is to analyze the provided excerpts and deliver answers that are:
- **Accurate**, based solely on the given excerpts
- **Clear and concise**, tailored for business and finance professionals
- **Insightful**, especially when multiple sources contribute to the answer

### Context:
{context}

### User Question:
"{question}"

### Instructions:
- Use **only** the information provided in the excerpts. Do **not** rely on external knowledge or assumptions.
- Structure your response logically. If multiple excerpts are relevant, **synthesize the information coherently**.
- For questions involving **figures, dates, or performance**, consider presenting key details in a **markdown table**.
- Ensure all financial or business terms are **explained clearly**, and highlight comparisons or trends when relevant.
- Avoid verbosity. Focus on clarity, precision, and completeness.
- If no relevant information is found, respond with:  
    `*No Data Available*`

### Output Format:
Provide the final answer in **Markdown**.  
End with a list of sources used in this exact format:
Sources:
Source1: [file name], [page_number: X]
Source2: [file name], [page_number: Y]
"""

# === Final Function ===
def answer_financial_question(question: str) -> str:
    try:
        vector_store = load_vectorstore()
        # If vector store does not exist, create it from documents
        if not vector_store:
            print("🆕 Creating FAISS index...")
            docs = extract_documents_from_folder(FOLDER_PATH)
            if not docs:
                return "**Error:** No documents found to create vector store."
            # Chunking is handled in process_txt/process_pdf
            vector_store = FAISS.from_documents(docs, embedding_model)
            vector_store.save_local(FAISS_PATH)
            print("✅ FAISS index created and saved.")

        # Step 1: Decompose question into sub-questions
        subquery_chain = subquery_prompt | llm
        sub_questions_output = subquery_chain.invoke({"question": question})
        sub_questions = sub_questions_output.content.strip().split("\n")
        sub_questions = [q.strip("- ").strip() for q in sub_questions if q.strip()]

        # Step 2: Perform similarity search for each sub-question
        all_context = []
        for sub_q in sub_questions:
            try:
                docs = vector_store.similarity_search(sub_q, k=3)
                all_context.extend(docs)
            except Exception as e:
                print(f"❌ Search failed for sub-question '{sub_q}': {e}")

        # Step 3: Deduplicate
        unique_docs = {}
        for doc in all_context:
            key = (doc.metadata.get("filename", ""), doc.page_content.strip())
            unique_docs[key] = doc
        context_docs = list(unique_docs.values())

        # Step 4: Rerank
        all_context_reranked = rerank_documents(question, context_docs, embedding_model)

        # Step 5: Run final chain
        final_prompt = ChatPromptTemplate.from_template(main_prompt_template)
        chain = create_stuff_documents_chain(llm, final_prompt, document_variable_name="context")
        result = chain.invoke({
            "context": all_context_reranked,
            "question": question
        })
        return result

    except Exception as e:
        traceback.print_exc()
        return f"❌ An error occurred: {e}"



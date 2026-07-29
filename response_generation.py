import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_rag_with_sources(file_path, query):
    reader = PdfReader(file_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
            
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    document_chunks = splitter.create_documents([raw_text])
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(document_chunks, embeddings_model)
    
    
    retrieved_docs = vector_store.similarity_search_with_score(query, k=2)
    
    print(f"Query: {query}\n")
    print("--- GENERATED RESPONSE & SOURCES ---")
    
    for i, (doc, score) in enumerate(retrieved_docs):
        print(f"\n[Source {i+1}] (Relevance Score: {score:.4f})")
        print(f"Content snippet: {doc.page_content.strip()}")

#test bolumu
if __name__ == "__main__":
    pdf_path = "test_file.pdf"
    query = "Who is Gregor Samsa?"
    run_rag_with_sources(pdf_path, query)
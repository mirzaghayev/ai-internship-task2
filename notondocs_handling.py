import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def query_with_fallback(file_path, query, threshold=1.2):
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
    

    retrieved_docs = vector_store.similarity_search_with_score(query, k=1)
    
    print(f"Query: '{query}'")
    
    if retrieved_docs:
        doc, score = retrieved_docs[0]
        if score > threshold:
            print("Answer: I am sorry, but this information is not on the docs.")
        else:
            print(f"Answer found based on docs (Score: {score:.4f}):")
            print(doc.page_content.strip())
    else:
        print("Answer: I am sorry, but this information is not on the docs.")

# yoxlama hissesi
if __name__ == "__main__":
    pdf_path = "test_file.pdf"
    
    # Test 1: Cavabi olan query
    query_valid = "Who is Gregor Samsa?"
    query_with_fallback(pdf_path, query_valid)
    
    print("-" * 50)
    
    # Test 2: Cavabi olmayan query
    query_invalid = "What is the capital city of Australia?"
    query_with_fallback(pdf_path, query_invalid)
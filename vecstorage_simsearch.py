
#this is just to ignore warnings
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(file_path):
    #evvelkilerde oldugu kimi chunklari cixardiriq
    reader = PdfReader(file_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
            
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    document_chunks = splitter.create_documents([raw_text])
    
    # cp 2 embedding model secimi
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # vector storage and indexing using FAISS 
    vector_store = FAISS.from_documents(document_chunks, embeddings_model)
    
    return vector_store

#test bolumu
if __name__ == "__main__":
    pdf_path = "test_file.pdf"
    print("Building FAISS vector store from document...")
    vector_store = build_vector_store(pdf_path)
    print("Vector store successfully built!")
    
    # burada query veririk ve similarity search edir
    query = "Who is Gregor Samsa?"
    print(f"\nPerforming similarity search for query: '{query}'")
    
    results = vector_store.similarity_search(query, k=2)
    
    for i, res in enumerate(results):
        print(f"\nMATCH {i+1}:")
        print(res.page_content)
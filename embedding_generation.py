from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

def generate_embeddings_for_chunks(file_path):
    # cp 1 deki kimi burda da extract edirik
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
    
    # Hugging face den yungul embedding modelini elave edirik
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # cixarilmis chunklardan embeddingler elave edirik
    texts = [chunk.page_content for chunk in document_chunks]
    embeddings = embeddings_model.embed_documents(texts)
    
    return document_chunks, embeddings

#test etmek ucun cp 1 de de istifade etdiyim eyni pdf fileni qeyd edirem ve embedding vector dimensionu print edirem
if __name__ == "__main__":
    pdf_path = "test_file.pdf"
    print("creating embeddings for chunks: ")
    chunks, chunk_embeddings = generate_embeddings_for_chunks(pdf_path)
    
    print(f"successfully generated {len(chunk_embeddings)} embeddings")
    print(f"dimension of single embedding vector: {len(chunk_embeddings[0])}")
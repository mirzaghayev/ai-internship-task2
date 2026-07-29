from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_and_chunk_pdf(file_path):
    #birinci olaraq her sehifeden metnleri cixardiriq
    reader = PdfReader(file_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    
    # Burada text splitterin size ve overlapini ayarlayiriq
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    
    # sonuncu olaraq metn chunklari yaradiriq
    document_chunks = splitter.create_documents([raw_text])
    return document_chunks

#yoxlama
if __name__ == "__main__":
    #file bura tam adresini yaziriq
    pdf_path = "test_file.pdf" 
    
    print("Loading and chunking PDF...")
    chunks = process_and_chunk_pdf(pdf_path)
    
    print(f"Successfully created {len(chunks)} chunks!\n")
    
    # ilk 2 chunki output edirik
    for i, chunk in enumerate(chunks[:2]):
        print(f"--- CHUNK {i+1} (Length: {len(chunk.page_content)}) ---")
        print(chunk.page_content)
        print("\n" + "="*40 + "\n")
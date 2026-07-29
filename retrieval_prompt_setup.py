import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

def setup_retrieval_and_prompt(file_path, query):
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
    
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    #sadece prompt template daxil edirik
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    # final formatlanmis prompt daxil edilmesi
    final_prompt = prompt_template.format(context=context, question=query)
    
    return final_prompt

# yoxlama bolumu:
if __name__ == "__main__":
    pdf_path = "test_file.pdf"
    query = "Who is Gregor Samsa?"
    
    print("Setting up retrieval and formatting prompt...")
    prompt = setup_retrieval_and_prompt(pdf_path, query)
    
    print("\n--- GENERATED PROMPT ---")
    print(prompt)
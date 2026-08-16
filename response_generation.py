from retrieval_prompt_setup import setup_retrieval_and_prompt
from rag_utils import call_llm


def run_rag_with_sources(file_path, query, k=3):
    retrieved_docs, messages = setup_retrieval_and_prompt(file_path, query, k=k)

    # the actual generation step — this was missing before
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    answer = call_llm(user_prompt, system_prompt)

    print(f"Query: {query}\n")
    print("--- GENERATED RESPONSE ---")
    print(answer)

    print("\n--- SOURCES USED ---")
    for i, doc in enumerate(retrieved_docs, start=1):
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.strip().replace("\n", " ")[:120]
        print(f"[Chunk {i}] page {page}: {snippet}...")

    return answer


# test bolumu
if __name__ == "__main__":
    pdf_path = "test_file.pdf"
    query = "Who is Gregor Samsa?"
    run_rag_with_sources(pdf_path, query)
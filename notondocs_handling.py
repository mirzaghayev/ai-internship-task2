from response_generation import run_rag_with_sources


def query_with_fallback(file_path, query):
    """
    No distance-score threshold here on purpose — thresholds are fragile
    and easy to miscalibrate. Instead, the LLM itself is instructed
    (see rag_utils.SYSTEM_PROMPT) to say it doesn't know when the answer
    isn't in the retrieved context. This is what actually gets tested by
    a "hallucination trick" question with no answer in the docs.
    """
    return run_rag_with_sources(file_path, query)


# yoxlama hissesi
if __name__ == "__main__":
    pdf_path = "test_file.pdf"

    # Test 1: cavabi olan query
    query_valid = "Who is Gregor Samsa?"
    query_with_fallback(pdf_path, query_valid)

    print("-" * 50)

    # Test 2: cavabi olmayan query (should trigger the "I don't know" fallback)
    query_invalid = "What is the capital city of Australia?"
    query_with_fallback(pdf_path, query_invalid)
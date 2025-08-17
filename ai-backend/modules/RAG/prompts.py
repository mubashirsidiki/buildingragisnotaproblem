"""
AI prompts for the PDF RAG service.
"""


def get_rag_prompt(query: str, context: str) -> str:
    """
    Generate a RAG prompt for answering questions based on PDF content,
    allowing reasonable inference and graceful fallback to general knowledge.
    """
    return f"""
    Question: {query}

    Relevant information from the PDF:

    {context}

    Provide the most accurate and helpful answer you can, using the provided information as your main source. 
    You may make reasonable inferences or explain related details if they help clarify the answer, 
    but avoid introducing facts that clearly contradict the provided information. 

    If you cannot find enough relevant information in the provided content, 
    apologize and then answer the question based on your own general knowledge.

    If the user specifically requests information from your knowledge base, 
    call the appropriate function to retrieve it.
    """



def get_query_expansion_prompt(query: str) -> str:
    """Generate multiple perspective queries for better search coverage."""
    return f"""Generate 3 different search queries from different perspectives for: "{query}"

Each query should approach the topic differently but seek the same information.
Return only the queries, one per line, no explanations.

Example:
Original: "machine learning algorithms"
1. machine learning algorithms implementation
2. AI algorithm types and applications  
3. supervised unsupervised learning methods

Original: "{query}"
1.
2.
3.""" 
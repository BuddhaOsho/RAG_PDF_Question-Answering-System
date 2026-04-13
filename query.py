from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Load embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embeddings-3-large"
)

# Connect to existing Qdrant collection
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    embedding=embedding_model,
    collection_name="learning_rag"
)

# Take user query
user_query = input("Ask something: ")

# Retrieve relevant chunks
search_results = vector_db.similarity_search(query=user_query)

# Build context from retrieved documents
context = "\n\n".join(
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata.get('page_label')}\n"
    f"Source: {result.metadata.get('source')}"
    for result in search_results
)

# System prompt with grounded context
system_prompt = f"""
You are a helpful AI assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not present in the context, say you don't know.

Context:
{context}
"""

# Generate response
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
)

print("\nAnswer:\n")
print(response.choices[0].message.content)

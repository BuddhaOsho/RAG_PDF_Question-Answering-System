from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Path to the PDF file
pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load PDF document
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(documents)

# Create embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embeddings-3-large"
)

# Store embeddings in Qdrant
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("✅ Document indexing completed successfully.")

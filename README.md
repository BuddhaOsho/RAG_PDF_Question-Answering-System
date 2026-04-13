RAG PDF Question Answering System

A Retrieval‑Augmented Generation (RAG) system that enables question answering over PDF documents using vector search and LLMs.
Documents are indexed into a Qdrant vector database, and relevant chunks are retrieved at query time to generate grounded, context‑aware answers.

🚀 Features

📥 PDF Document Ingestion
✂️ Recursive text chunking for better context handling
🧠 Vector embeddings using OpenAI embeddings
🗄️ Qdrant vector database for semantic search
🔍 Similarity‑based retrieval
💬 LLM‑powered question answering
📄 Answers include page content and source metadata
🐳 Dockerized vector database setup

Architecture Overview:

PDF Documents
     ↓
Text Chunking
     ↓
Vector Embeddings
     ↓
Qdrant Vector DB (Docker)
     ↓
Similarity Search
     ↓
LLM with Retrieved Context
     ↓
Final Answer

Tech Stack


Tech Stack:

Language: Python
LLM: OpenAI
Framework: LangChain
Vector DB: Qdrant
Embeddings: OpenAI Embeddings
Containerization: Docker & Docker Compose
Environment Management: python‑dotenv

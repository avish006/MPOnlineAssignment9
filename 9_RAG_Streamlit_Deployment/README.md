# 📚 Conversational RAG Chatbot with Multi-PDF Upload

> **This project is developed as a Capstone Project for MPONLINE.**

A powerful, conversational Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and ChromaDB. Upload multiple PDF documents and have an intelligent, context-aware conversation about their contents.

---

## ✨ Features

- **Multi-PDF Upload**: Upload and process multiple PDF documents simultaneously
- **Recursive Text Splitting**: Smart document chunking using LangChain's `RecursiveCharacterTextSplitter` for optimal retrieval
- **Semantic Search**: Embeddings powered by `all-MiniLM-L6-v2` (HuggingFace) stored in ChromaDB
- **Conversational Memory**: Full chat history awareness — ask follow-up questions naturally
- **Gemini LLM**: Powered by Google's Gemini model via the Gemini API
- **Clean Streamlit UI**: Simple, intuitive chat interface with a sidebar for document management

---

## 🏗️ Architecture

```
PDFs → Text Extraction (PyPDF2)
     → Chunking (RecursiveCharacterTextSplitter)
     → Embedding (all-MiniLM-L6-v2)
     → Vector Store (ChromaDB)
     → History-Aware Retriever (LCEL)
     → LLM (Gemini via Google AI)
     → Streamlit Chat UI
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Google Gemini API Key ([Get one here](https://aistudio.google.com/))

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/RAG_Streamlit.git
cd RAG_Streamlit
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API Key
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Run the application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
RAG_Streamlit/
├── app.py              # Streamlit UI — chat interface & sidebar
├── utils.py            # Core RAG logic (extraction, chunking, embedding, chain)
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed to git)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **UI Framework** | Streamlit |
| **LLM** | Google Gemini (via `langchain-google-genai`) |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Database** | ChromaDB |
| **Orchestration** | LangChain (LCEL) |
| **PDF Parsing** | PyPDF2 |

---

## 🔒 Security Note

Never commit your `.env` file or any file containing API keys. The `.gitignore` in this project already excludes it.

---

## 📄 License

This project is for educational purposes as part of the **MPONLINE Capstone Project**.

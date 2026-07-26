import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def get_text_chunks(text):
    """Splits the given text into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Embeds text chunks and initializes a ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store vectors in a local directory
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

def get_conversational_chain(vector_store, api_key: str = None):
    """Creates a conversational RAG chain with chat history awareness using LCEL."""
    # Use the passed-in key, or fall back to the environment variable
    google_api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
    if not google_api_key or google_api_key.strip() == "":
        raise ValueError("Please provide a valid Google Gemini API key.")

    llm = ChatGoogleGenerativeAI(
        google_api_key=google_api_key,
        model="gemini-3-flash-preview",
        temperature=0.3
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # --- Prompt 1: Rephrase user question considering chat history ---
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate a standalone question that can be understood without the history. "
         "Do NOT answer it — just rephrase if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # --- Prompt 2: Answer based on retrieved context ---
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for question-answering tasks. "
         "Use the retrieved context below to answer the question. "
         "If you don't know the answer, say so. Keep answers concise.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_standalone_question(data):
        """Rephrase the question using chat history."""
        # If no history, just use input as-is
        if not data.get("chat_history"):
            return data["input"]
        chain = contextualize_q_prompt | llm | StrOutputParser()
        return chain.invoke(data)

    # Full RAG chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(get_standalone_question)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["standalone_question"])))
        )
        | RunnablePassthrough.assign(
            answer=qa_prompt | llm | StrOutputParser()
        )
    )

    return rag_chain

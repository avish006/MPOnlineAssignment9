import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from utils import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "google_api_key" not in st.session_state:
        # Pre-fill from .env if available (for local dev), else empty
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")

def main():
    st.title("Conversational RAG with Multi-PDF Upload 📚")

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # --- API Key Input ---
        api_key_input = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.google_api_key,
            type="password",
            placeholder="Paste your Gemini API key here...",
            help="Get your free API key from https://aistudio.google.com/",
        )

        # Update session state if user changed the key
        if api_key_input != st.session_state.google_api_key:
            st.session_state.google_api_key = api_key_input
            # Reset everything if the key changes
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.messages = []

        if not st.session_state.google_api_key:
            st.warning("Please enter your Google Gemini API key to get started.")
            st.markdown(
                "🔑 [Get a free API key here](https://aistudio.google.com/)",
                unsafe_allow_html=False,
            )
            st.stop()

        st.divider()

        # --- Document Upload ---
        st.subheader("📄 Document Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Process Documents", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDFs.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)

                    # Reset chat when new docs are processed
                    st.session_state.chat_history = []
                    st.session_state.messages = []

                    st.success(f"✅ Processed {len(pdf_docs)} PDF(s)! Start chatting below.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.vector_store is None:
            st.error("Please upload and process documents first.")
            return

        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                # Pass the API key from session state into the chain
                rag_chain = get_conversational_chain(
                    st.session_state.vector_store,
                    api_key=st.session_state.google_api_key
                )

                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history,
                })

                answer = response["answer"]

                # Update LangChain chat history
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer),
                ])

                # Show assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

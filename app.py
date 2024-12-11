# Import necessary libraries
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WikipediaLoader
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import wikipedia
import numpy as np


# Configure Google Generative AI
genai.configure(api_key="your_key")  # Replace with your own API Key
model = genai.GenerativeModel('gemini-1.5-flash')  # Replace model with appropriate name if needed

# Wrapper for SentenceTransformer
class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # Embed a list of documents
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        # Embed a single query
        return self.model.encode(text, convert_to_numpy=True)

# Initialize embedding model and wrapper
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
embedding_wrapper = SentenceTransformerWrapper(sentence_transformer)


# Initialize a Chroma vector store
pdf_db = Chroma(embedding_function=embedding_wrapper, persist_directory="pdf_vector_store")
wiki_db = Chroma(embedding_function=embedding_wrapper, persist_directory="wiki_vector_store")

# Helper function to split and embed text
def add_to_vectorstore(text, db):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = [sentence_transformer.encode(chunk) for chunk in chunks]
    for i, chunk in enumerate(chunks):
        db.add_texts([chunk], embeddings=[embeddings[i]])

# Extract text from PDF
def pdf_to_text(uploaded_pdf):
    """Convert uploaded PDF to plain text."""
    pdf_reader = PdfReader(uploaded_pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text.strip()

# Query AI model
def query_with_google_ai(context, query):
    """Query Google Generative AI model."""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

# Retrieve and generate response
def retrieve_and_generate(db, query):
    """Retrieve relevant context and generate a response."""
    try:
        results = db.similarity_search(query, k=5)  # Retrieve top 5 relevant chunks
        context = "\n".join([doc.page_content for doc in results])
        return query_with_google_ai(context, query)
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize Streamlit app
st.title("📄 Interactive Chat with PDF or Wikipedia")
st.write("Upload a PDF or select a Wikipedia topic to start asking questions!")

# Create tabs for PDF chat and Wikipedia query
tabs = st.tabs(["Chat with PDF", "Chat with Wikipedia"])

# PDF chat tab
with tabs[0]:
    st.subheader("📄 Chat with PDF")
    # Initialize session state for chat history
    if "pdf_past" not in st.session_state:
        st.session_state["pdf_past"] = []
    if "pdf_generated" not in st.session_state:
        st.session_state["pdf_generated"] = []

    # PDF upload
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        st.success("PDF uploaded successfully!")
        pdf_text = pdf_to_text(uploaded_pdf)
        add_to_vectorstore(pdf_text, pdf_db)
        st.write("PDF content has been indexed for search.")

    # Containers for chat history and input
    response_container = st.container()
    container = st.container()

    # Chat input form
    with container:
        with st.form(key="pdf_chat_form", clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask something about your document...")
            submit_button = st.form_submit_button(label="Send")
            if submit_button and user_input:
                answer = retrieve_and_generate(pdf_db, user_input)
                st.session_state["pdf_past"].append(user_input)
                st.session_state["pdf_generated"].append(answer)

    # Display chat history
    if st.session_state["pdf_generated"]:
        with response_container:
            for i, (user_message, bot_message) in enumerate(zip(st.session_state["pdf_past"], st.session_state["pdf_generated"])):
                message(user_message, is_user=True, key=f"{i}_pdf_user", avatar_style="big-smile")
                message(bot_message, key=f"{i}_pdf", avatar_style="thumbs")

# Wikipedia chat tab
with tabs[1]:
    st.subheader("🌍 Chat with Wikipedia")
    # Initialize session state for chat history
    if "wiki_past" not in st.session_state:
        st.session_state["wiki_past"] = []
    if "wiki_generated" not in st.session_state:
        st.session_state["wiki_generated"] = []

    # Wikipedia Topic selection
    st.subheader("Select a Wikipedia Topic")
    search_query = st.text_input("Start typing to search for a topic:")

    if search_query:
        try:
            # Search Wikipedia for the query (limit to top 5 results)
            search_results = wikipedia.search(search_query, results=5)
            if search_results:
                topic = st.selectbox("Select a topic:", search_results)

                if topic:
                    wiki_loader = WikipediaLoader(query=topic, load_max_docs=5)
                    docs = wiki_loader.load()
                    context = "\n".join([doc.page_content for doc in docs])
                    add_to_vectorstore(context, wiki_db)

                    st.write("Retrieved content has been indexed for search.")

                # Containers for chat history and input
                response_container = st.container()
                container = st.container()

                # Chat input form
                with container:
                    with st.form(key="wiki_chat_form", clear_on_submit=True):
                        user_input = st.text_input("Ask a question:", placeholder="Ask something about the topic...")
                        submit_button = st.form_submit_button(label="Send")
                        if submit_button and user_input:
                            answer = retrieve_and_generate(wiki_db, user_input)
                            st.session_state["wiki_past"].append(user_input)
                            st.session_state["wiki_generated"].append(answer)

                # Display chat history
                if st.session_state["wiki_generated"]:
                    with response_container:
                        for i, (user_message, bot_message) in enumerate(zip(st.session_state["wiki_past"], st.session_state["wiki_generated"])):
                            message(user_message, is_user=True, key=f"{i}_wiki_user", avatar_style="big-smile")
                            message(bot_message, key=f"{i}_wiki", avatar_style="thumbs")

            else:
                st.write("No results found. Try a different search term.")
        except wikipedia.exceptions.HTTPTimeoutError:
            st.write("Request to Wikipedia timed out. Please try again later.")
        except wikipedia.exceptions.ConnectionError:
            st.write("Network connection error. Please check your internet connection.")


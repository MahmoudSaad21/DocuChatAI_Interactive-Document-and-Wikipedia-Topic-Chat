# Import necessary libraries
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from streamlit_chat import message
import wikipedia
import numpy as np


# Configure Google Generative AI
genai.configure(api_key="your_key")  # Replace with your own API Key
model = genai.GenerativeModel('gemini-1.5-flash')  # Replace model with appropriate name if needed

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
def pdf_to_text(uploaded_pdf):
    """Convert uploaded PDF to plain text."""
    pdf_reader = PdfReader(uploaded_pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text.strip().split("\n")

# Query AI model
def query_with_google_ai(context, query):
    """Query Google Generative AI model."""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

# Conversational chat handler
def conversational_chat(context, query):
    try:
        response = query_with_google_ai(context, query)
        return response
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
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "pdf_context" not in st.session_state:
        st.session_state["pdf_context"] = ""

    # PDF upload (now on the main page)
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        st.success("PDF uploaded successfully!")
        pdf_lines = pdf_to_text(uploaded_pdf)
        st.session_state["pdf_context"] = " ".join(pdf_lines)  # Use first 20 lines as context

        st.subheader("PDF Preview")
        st.write("\n".join(pdf_lines[:5]))  # Display first 5 lines as preview

    # Containers for chat history and input
    response_container = st.container()
    container = st.container()

    # Chat input form
    with container:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask something about your document...")
            submit_button = st.form_submit_button(label="Send")
            if submit_button and user_input:
                context = st.session_state["pdf_context"]
                answer = conversational_chat(context, user_input)
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(answer)

    # Display chat history
    if st.session_state["generated"]:
        with response_container:
            for i, (user_message, bot_message) in enumerate(zip(st.session_state["past"], st.session_state["generated"])):
                message(user_message, is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(bot_message, key=str(i), avatar_style="thumbs")


# Wikipedia chat tab
with tabs[1]:
    st.subheader("🌍 Chat with Wikipedia")
    
    # Initialize session state for chat history
    if "wiki_past" not in st.session_state:
        st.session_state["wiki_past"] = []
    if "wiki_generated" not in st.session_state:
        st.session_state["wiki_generated"] = []
    if "wiki_context" not in st.session_state:
        st.session_state["wiki_context"] = ""

    # Wikipedia Topic selection
    st.subheader("🔎 Select a Wikipedia Topic")
    topic = ""
    # Let the user input a search term
    search_query = st.text_input("Start typing to search for a topic:")

    if search_query:
        try:
            # Search Wikipedia for the query (limit to top 5 results)
            search_results = wikipedia.search(search_query, results=5)
            if search_results:
                # Dynamically update selectbox based on search results
                topic = st.selectbox("Select a topic:", search_results)
                
                # If a topic is selected, load the content
                if topic:
                    wiki_loader = WikipediaLoader(query=topic, load_max_docs=5)
                    docs = wiki_loader.load()
                    context = "\n".join([doc.page_content for doc in docs])
                    st.session_state["wiki_context"] = context

                    st.write("Retrieved context from Wikipedia:")
                    st.write(context[:500])  # Display first 500 characters of the context

                    # Containers for chat history and input
                    response_container = st.container()
                    container = st.container()

                    # Chat input form
                    with container:
                        with st.form(key="wiki_chat_form", clear_on_submit=True):
                            user_input = st.text_input("Ask a question:", placeholder="Ask something about the topic...")
                            submit_button = st.form_submit_button(label="Send")
                            if submit_button and user_input:
                                context = st.session_state["wiki_context"]
                                answer = conversational_chat(context, user_input)
                                st.session_state["wiki_past"].append(user_input)
                                st.session_state["wiki_generated"].append(answer)

                    # Display chat history for Wikipedia
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

    

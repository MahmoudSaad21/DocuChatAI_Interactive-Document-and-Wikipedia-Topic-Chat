# DocuChat AI: Interactive Document and Wikipedia Topic Chat

This project includes a Streamlit-based application (`app.py`) and a Jupyter Notebook (`LLM_Experiments_and_Text_Processing.ipynb`) designed for engaging users in interactive conversations with uploaded PDFs or Wikipedia topics.The app introduces a vector store for efficient similarity searches and leverages embedding models for robust context understanding. Additionally, a Jupyter Notebook provides experimentation insights with Language Models.
.

---

## Files in the Repository

### `app.py`
This Python script powers the Streamlit application. Key functionalities include:
- **PDF Chat**:
  - Extracts text from PDFs using `PyPDF2`.
  - Embeds text chunks into a vector store using `Chroma` for similarity-based search.
  - Allows natural language querying through Google Generative AI.

- **Wikipedia Chat**:
  - Fetches topic content from Wikipedia using `WikipediaLoader`.
  - Stores content in a vector store for fast retrieval and contextual querying.

- **Vector Store Implementation**:
  - Introduced `Chroma` as a vector database for managing embeddings.
  - Supports similarity search to fetch top-relevant text chunks for AI-driven query responses.

- **Embeddings**:
  - Uses a wrapper class around `SentenceTransformer` for embedding documents and queries.

### `LLM_Experiments_and_Text_Processing.ipynb`
This Jupyter Notebook contains:
- Experimentation with Language Models (LLMs).
- Examples of integrating embeddings and AI-driven query systems.
- Demonstrations of text processing and conversational AI techniques.

---

## Features
- **PDF Chat**: Upload a PDF, index its content, and query it naturally.
- **Wikipedia Chat**: Search for Wikipedia topics, index the retrieved content, and engage in a Q&A.
- **Vector-Based Search**: Efficient context retrieval using similarity search in vector stores.
- **Enhanced Contextual Responses**: Combines embeddings and generative AI for accurate and insightful responses.
- **Interactive Interface**: Streamlined chat history and query handling using Streamlit's session state.

---

## Technical Details

### Libraries and Tools Used
- **Streamlit**: Framework for creating the web-based interface.
- **Chroma**: Vector store for similarity-based text retrieval.
- **Google Generative AI**: Natural language generation and contextual understanding.
- **SentenceTransformers**: Model for text embeddings.
- **PyPDF2**: Extracts text from PDF files.
- **Wikipedia API**: Searches and retrieves content from Wikipedia.
- **LangChain**: Tools for text splitting and workflow creation.
- **Numpy**: Efficient handling of numerical data.

### System Workflow
1. **Input Handling**:
   - PDFs are uploaded, and their content is extracted.
   - Wikipedia topics are searched and selected by the user.
2. **Text Indexing**:
   - PDFs and Wikipedia topics are split into chunks and embedded into a vector store.
3. **Similarity Search**:
   - Retrieves top-k relevant chunks from the vector store based on query similarity.
4. **AI Query**:
   - Google Generative AI is prompted with context and user queries.
   - The response is displayed in a conversational format.
5. **Chat Interface**:
   - Displays past queries and AI responses in an intuitive layout.

---

## Installation

### Requirements
- Python 3.8 or higher
- The required libraries are listed in `requirements.txt`.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/MahmoudSaad21/DocuChatAI_Interactive-Document-and-Wikipedia-Topic-Chat.git
   cd DocuChatAI_Interactive-Document-and-Wikipedia-Topic-Chat
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### Chat with PDFs
1. Upload a PDF file.
2. Ask questions about the document's content.

### Chat with Wikipedia
1. Search for a topic.
2. Select the relevant Wikipedia entry from the results.
3. Ask questions about the topic.

---

## Demonstration
Watch a video demonstration here:

https://github.com/user-attachments/assets/ac9a0acc-32c1-40c7-a2d2-96b9658f586f


---

## Contributing
Contributions are welcome! Please create a pull request with your changes.

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://developers.generative.ai/)
- [Wikipedia](https://www.wikipedia.org/)
- [LangChain](https://www.langchain.com/)
- [Chroma](https://www.trychroma.com/)

# DocuChat AI: Interactive Document and Wikipedia Topic Chat

This project includes a Streamlit-based application (`app.py`) and a Jupyter Notebook (`LLM_Experiments_and_Text_Processing.ipynb`) designed for engaging users in interactive conversations with uploaded PDFs or Wikipedia topics. The app leverages Google Generative AI and other advanced libraries to process and respond to user queries based on contextual information from the provided sources.

---

## Files in the Repository

### `app.py`
This Python script powers the Streamlit application. Key functionalities include:
- **PDF Chat**:
  - Extracts text from uploaded PDF files using `PyPDF2`.
  - Integrates Google Generative AI for querying extracted content.
- **Wikipedia Chat**:
  - Retrieves content from Wikipedia using `wikipedia` and `WikipediaLoader`.
  - Provides an interactive chat experience with selected Wikipedia topics.
- **Embeddings**:
  - Uses `SentenceTransformer` for creating embeddings of text data.
- **Streamlit Chat**:
  - Implements a dynamic chat interface with session-based history management.

### `LLM_Experiments_and_Text_Processing.ipynb`
This Jupyter Notebook contains:
- Experimentation with Language Models (LLMs).
- Examples of integrating embeddings and AI-driven query systems.
- Demonstrations of text processing and conversational AI techniques.

---

## Features
- **Chat with PDFs**: Upload a PDF document, extract text, and interactively query the content.
- **Chat with Wikipedia**: Search, select, and chat about Wikipedia topics in real-time.
- **Embeddings and AI Models**: Utilizes advanced text embeddings and Generative AI for enhanced query responses.
- **User-Friendly Interface**: Intuitive design powered by Streamlit for seamless interactions.

---

## Technical Details

### Libraries and Tools Used
- **Streamlit**: Framework for building interactive web applications.
- **Google Generative AI**: Provides natural language understanding and generation capabilities.
- **SentenceTransformers**: Embeddings for text comparison and search.
- **PyPDF2**: Extracts text from PDF files.
- **Wikipedia API**: Retrieves content from Wikipedia for processing.
- **LangChain Community**: Supports text splitting and prompt management.
- **Numpy**: Efficient handling of numerical data.

### System Workflow
1. **Input Handling**:
   - PDFs are uploaded, and their content is extracted.
   - Wikipedia topics are searched and selected by the user.
2. **Text Processing**:
   - Extracted or retrieved text is split and embedded.
   - Context is created for AI-based query handling.
3. **AI Query**:
   - Google Generative AI is prompted with context and user queries.
   - The response is displayed in a conversational format.
4. **Chat Interface**:
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
[Demo Video](#) *(Replace with a YouTube or hosted video link)*

---

## Contributing
Contributions are welcome! Please create a pull request with your changes.


---

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://developers.generative.ai/)
- [Wikipedia](https://www.wikipedia.org/)


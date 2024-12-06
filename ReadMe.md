
# Yo'l Harakati Qoidalari Chat Bot

This project is a **Streamlit application** that allows users to interact with a chatbot trained on a PDF document containing traffic rules (Yo'l Harakati Qoidalari). The bot uses OpenAI's GPT model to answer user queries based on the document or provide general guidance on traffic-related questions.

## Features

- **PDF Parsing and Vector Store Creation**:
  - The application uses `PyPDFLoader` to load a PDF document and splits it into manageable chunks using `RecursiveCharacterTextSplitter`.
  - A vector store is created using FAISS and OpenAI embeddings to enable fast and efficient similarity-based search.

- **User-Friendly Chat Interface**:
  - Users can input their queries in a text box.
  - The bot responds with information retrieved from the PDF or general knowledge about traffic rules.

- **Customizable Design**:
  - Styled with CSS for a polished look, including customized input boxes, buttons, and response formatting.

- **Session Management**:
  - Chat history is maintained during the session for a seamless user experience.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8 or later
- Required Python packages (see below)

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run chatbot.py
   ```

2. Open the application in your browser (Streamlit will provide a local URL).

3. Enter your OpenAI API key in the input box to start the session.

4. Ask questions about the traffic rules, and the bot will provide answers based on the PDF or general traffic rules knowledge.

## File Structure

- `chatbot.py`: Main application script.
- `yhq.pdf`: PDF file containing traffic rules (to be added).
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Technologies Used

- **Streamlit**: For creating the user interface.
- **LangChain**: For PDF document loading and text processing.
- **FAISS**: For efficient similarity search.
- **OpenAI API**: For generating chatbot responses.
- **Python**: Core programming language.

## Customization

- **PDF Document**: Replace `yhq.pdf` with any PDF file to customize the knowledge base.
- **Styling**: Modify the CSS in the `st.markdown()` block to change the design.


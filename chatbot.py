import openai
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# OpenAI API kalitini olish
def get_openai_api_key():
    return st.text_input("OpenAI API kalitini kiriting", type="password")

# Matnni bo'lish va tozalash funksiyasi
def split_and_clean_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    cleaned_documents = []
    for doc in documents:
        # Har bir hujjatni bo'laklarga bo'lish
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            cleaned_documents.append(chunk.strip())
    return cleaned_documents

# PDF va vektor do'konni yuklash
@st.cache_resource
def load_pdf_and_create_vector_store(file_path, api_key, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Hujjatlarni bo'lish va tozalash
    cleaned_documents = split_and_clean_documents(documents, chunk_size, chunk_overlap)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(cleaned_documents, embeddings)
    return vector_store

# Foydalanuvchi savolini vektor do'kondan qidirish
def get_relevant_text(vector_store, query, top_k=5):
    results = vector_store.similarity_search(query, k=top_k)
    relevant_text = "\n".join([res.page_content for res in results])
    return relevant_text

# ChatGPT javobini olish
def get_response(api_key, relevant_text, query):
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": "Quyidagi faylga asoslangan holda savolga javob ber. Agar fayldagi ma'lumotlardan tashqari savol berilsa, sizga yo'l harakati qoidalari haqidagi savollarizga yordam bera olaman deb javob qaytar. Shuningdek O'zbekiston yo'l harakatiga oid barcha savollarga javob ber."},
        {"role": "assistant", "content": relevant_text},
        {"role": "user", "content": query},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message["content"]

# Streamlit ilovasi
def main():
    st.set_page_config(page_title="Yo'l Harakati Qoidalari Chat", layout="wide")
    
    # Fontlar va ranglarni o'zgartirish
    st.markdown("""
        <style>
            body {
                font-family: 'Verdana', sans-serif;
                background-color: #fafafa;
                color: #444;
            }
            h1 {
                color: #3E4A59;
                font-size: 36px;
                text-align: center;
                margin-top: 20px;
                font-weight: bold;
            }
            .stTextInput input {
                background-color: black;
                border-radius: 8px;
                border: 1px solid #ddd;
                padding: 12px;
                width: 100%;
                box-sizing: border-box;
                font-size: 16px;
                font-family: 'Verdana', sans-serif;
            }
            .stButton button {
                background-color: #5e4b8b;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                font-size: 16px;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: black;
            }
            .response {
                font-size: 16px;
                background-color: yellow;
                border-radius: 8px;
                padding: 12px;
                margin-top: 10px;
                margin-bottom: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .user-message {
                text-align: left;
                color: #0077b6;
                font-weight: bold;
            }
            .assistant-message {
                text-align: right;
                color: red;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Yo'l Harakati Qoidalari haqidagi savollarga javob beradigan bot")

    # Foydalanuvchidan API kalitini olish
    api_key = get_openai_api_key()
    if not api_key:
        st.warning("Iltimos, OpenAI API kalitini kiriting.")
        return

    file_path = "yhq.pdf"  # PDF faylni ko'rsating

    if "vector_store" not in st.session_state:
        with st.spinner("PDF yuklanmoqda..."):
            vector_store = load_pdf_and_create_vector_store(
                file_path, api_key, chunk_size=1000, chunk_overlap=200
            )
            st.session_state.vector_store = vector_store

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Savolingizni kiriting:")
    if user_query:
        with st.spinner("Javob tayyorlanmoqda..."):
            relevant_text = get_relevant_text(st.session_state.vector_store, user_query, top_k=5)
            if relevant_text.strip():
                response = get_response(api_key, relevant_text, user_query)
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                response = "PDF faylda ushbu savolga tegishli ma'lumot topilmadi."
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Chat tarixini ko'rsatish
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"<div class='user-message'>{chat['content']}</div>", unsafe_allow_html=True)
        elif chat["role"] == "assistant":
            st.markdown(f"<div class='assistant-message'>{chat['content']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

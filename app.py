import streamlit as st
from utils import load_file, embed_documents, get_vectorstore
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

st.set_page_config(page_title="Book Chat AI", layout="wide")

st.title("📚 Book Chat AI - Powered by Ollama")

menu = ["Embed Books", "Chat with AI"]
choice = st.sidebar.radio("Select Mode", menu)

categories = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]

if choice == "Embed Books":
    st.subheader("📤 Upload and Embed a Book")
    
    uploaded_file = st.file_uploader("Upload a book", type=["pdf", "txt", "docx"])
    new_category = st.text_input("Or create a new category")
    
    category = new_category if new_category else st.selectbox("Select category", categories)

    if uploaded_file and category:
        save_path = os.path.join("data", category)
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to {file_path}")

        with st.spinner("Embedding..."):
            docs = load_file(file_path)
            embed_documents(category, docs)
            st.success("Embedding done!")

elif choice == "Chat with AI":
    st.subheader("💬 Chat with AI Based on Category")

    category = st.selectbox("Select category to chat with", categories)

    if st.button("Initialize Chat"):
        vectorstore = get_vectorstore(category)
        llm = Ollama(model="phi3")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        st.session_state.qa_chain = qa_chain

    if "qa_chain" in st.session_state:
        user_question = st.text_input("Ask a question")
        if user_question:
            output = st.session_state.qa_chain.invoke({"query": user_question})
            st.markdown(f"**AI:** {output['result']}")

            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(output['source_documents']):
                    st.markdown(f"**Doc {i+1}:**")
                    st.write(doc.page_content[:500] + "...")


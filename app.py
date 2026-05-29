import streamlit as st 
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq 

load_dotenv()


st.title("📄 SMART Q&A Chatbot")

uploaded_file = st.file_uploader("Upload the File",type="pdf")
question = st.text_input("whats your Querry about the file\n")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)
        chunks = splitter.split_documents(pages)
        print(f"The length of the chunks :",len(chunks))

        embeddings = HuggingFaceEndpointEmbeddings(
        model = "sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token = os.getenv("HF_TOKEN"))
        
        vectorstore = Chroma.from_documents(documents = chunks,embedding = embeddings,persist_directory = "./chroma_db" )

        llm = ChatGroq(model="llama-3.3-70b-versatile",api_key= os.getenv("GROQ_API_KEY"),temperature = 0)
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Keep the answer concise.\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            ("human","{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm,prompt)
        chain = create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k":3}),
            question_answer_chain
        )

        with st.spinner("Thinking..."):
            answer = chain.invoke({"input": question})
            st.success(answer["answer"])
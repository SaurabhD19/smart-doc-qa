import streamlit as st 
import tempfile
import os 
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
load_dotenv()

def load_documents(file_path,File_extension):
    if File_extension == "pdf":
        loader  = PyPDFLoader(file_path)
    elif File_extension == "csv":
        loader  = CSVLoader(file_path)
    elif File_extension == "docx":
        loader  = Docx2txtLoader(file_path)
    elif File_extension == "txt":
        loader  = TextLoader(file_path)
    elif File_extension == "xlsx":
        loader  = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported File Type {File_extension}")
    return loader.load()
    

st.title("📄 SMART Q&A Chatbot")

Model_name = (
    "sentence-transformers/all-MiniLM-L6-v2" if os.getenv("STREAMLIT_CLOUD")
    else "sentence-transformers/all-mpnet-base-v2"
)


uploaded_file = st.file_uploader("Upload the File",type=["pdf","csv","docx","txt","xlsx"])
question = st.text_input("whats your Querry about the file\n")

if uploaded_file and question:
    File_size_mb = uploaded_file.size / (1024 * 1024)
    if File_size_mb > 10 :
        st.error(f"File too large ({File_size_mb:.1f}MB). Please upload files under 10MB.")
        st.stop()
    File_extension = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False,suffix=f".{File_extension}") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    pages = load_documents(tmp_path,File_extension)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    chunks = splitter.split_documents(pages)
    print(f"The number of the chunks of the file is :",len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name = Model_name)
        
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
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from dotenv import load_dotenv 
import os
load_dotenv()


loader = PyPDFLoader("data/Saurabh_Dubey.pdf")
page = loader.load()




splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 100)

chunks = splitter.split_documents(page)
print(f"Total Length :",len(chunks))




embeddings = HuggingFaceEndpointEmbeddings(
    model = "sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token = os.getenv("HF_TOKEN"),
)


vectorstore = Chroma.from_documents(
    documents =chunks,
    embedding =  embeddings,
    persist_directory = "./chroma_db"
)



query = "what is the document about?"
result = vectorstore.similarity_search(query,k=3)

for r in result:
    print(r.page_content)
    print("....")


print("successfully stored in ChromaDB..!")
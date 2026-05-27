from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv 
import os
load_dotenv()


# Loading the PDF content 
loader = PyPDFLoader("data/Saurabh_Dubey.pdf")
page = loader.load()



# Split the text using RecursiveCharacterTextSplitter 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 0)

chunks = splitter.split_documents(page)
print(f"Total Length :",len(chunks))



#Using Huggingface Embeddings for Embedding
embeddings = HuggingFaceEndpointEmbeddings(
    model = "sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token = os.getenv("HF_TOKEN"),
)

#storing the chunks and embeddings in Chroma Database
vectorstore = Chroma.from_documents(
    documents =chunks,
    embedding =  embeddings,
    persist_directory = "./chroma_db"
)


# Querry to Search in Knowledgebase
query = input("enter the Query :\n")
result = vectorstore.similarity_search(query,k=1)

for r in result:
    print(r.page_content)
    print("....")



## Loading the Saved ChromaDB
vectorstore = Chroma(
    persist_directory = "./chroma_db",
    embedding_function = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        task = "feature-extraction",
        huggingfacehub_api_token = os.getenv("HF_TOKEN"),
))

querry2 = input("enter the Second querry on the loaded Database \n")
result2 = vectorstore.similarity_search(querry2,k=1)

for r in result2:
    print(r.page_content)
    print(".....")




# Smart Q&A Doc
![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.3-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)


"A document Answering System built with RAG Architecture — upload any document and get accurate answers instantly"



## 🌐 Live Demo
👉 [Try it here]()

---

## ⚙️ How It Works
1. Upload a document (PDF, DOCX, CSV, TXT, XLSX)
2. Document is split into chunks
3. Chunks are converted to embeddings
4. Embeddings stored in ChromaDB
5. Your question is matched against relevant chunks
6. Groq LLM generates answer from those chunks

---

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| Embeddings | HuggingFace `all-mpnet-base-v2`|
| Vector Database | ChromaDB |
| LLM | Groq (Llama 3.3 70B) |
| Framework | LangChain |
| UI | Streamlit |

---


## 📁 Project Structure
```
Smart-Doc-qa/
├── app.py
├── main.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md

```


## 💡 Files Supported
- PDF (.pdf)
- Word (.docx)
- Excel (.xlsx)
- CSV (.csv)
- Text (.txt)


## 🚀 Getting Started
### Installation
```bash
git clone https://github.com/SaurabhD19/smart-doc-qa.git
cd smart-doc-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the root folder:
```
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```
Get your free API keys:
- HuggingFace token → huggingface.co
- Groq API key → console.groq.com

### Run Locally
```bash
streamlit run app.py
```


## 🧠 Concepts Used 
- RAG (Retrival Augmented Generation)
- Text Chunking and Embeddings 
- Vector Similarity Search 
- LLM Prompt chaining



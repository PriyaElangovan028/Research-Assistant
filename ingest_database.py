import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import os

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

documents = []

# 🔥 LOAD PDF PROPERLY
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            text = page.get_text("text")

            # clean
            text = text.replace("\n", " ").strip()

            documents.append({
                "text": text,
                "page": page_num + 1,
                "source": file
            })

# splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = []

for doc in documents:
    split_texts = text_splitter.split_text(doc["text"])

    for chunk in split_texts:
        chunks.append({
            "page_content": chunk,
            "metadata": {
                "page": doc["page"],
                "source": doc["source"],
                "full_text": chunk
            }
        })

# convert to LangChain format
from langchain.schema import Document

docs = [
    Document(page_content=c["page_content"], metadata=c["metadata"])
    for c in chunks
]

ids = [str(uuid4()) for _ in docs]

vector_store.add_documents(documents=docs, ids=ids)
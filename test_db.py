from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS...")

db = FAISS.load_local(
    "vectordb",
    embeddings,
    allow_dangerous_deserialization=True
)

print("SUCCESS!")
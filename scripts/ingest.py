from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import os

docs = []

pdf_dir = "data"

print("Loading PDFs...")

for file in os.listdir(pdf_dir):

    if file.endswith(".pdf"):

        print(f"Reading {file}")

        loader = PyPDFLoader(
            os.path.join(pdf_dir, file)
        )

        docs.extend(loader.load())

print(f"\nPages loaded: {len(docs)}")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

batch_size = 500
progress_file = "progress.txt"

if os.path.exists(progress_file):

    with open(progress_file, "r") as f:
        last_batch = int(f.read().strip())

    print(f"\nResuming from Batch {last_batch + 1}")

else:

    last_batch = 0

    print("\nStarting from Batch 1")

db = None

if last_batch > 0 and os.path.exists("vectordb"):

    print("Loading existing FAISS index...")

    db = FAISS.load_local(
        "vectordb",
        embedding,
        allow_dangerous_deserialization=True
    )

start_index = last_batch * batch_size

for i in range(start_index, len(docs), batch_size):

    batch_docs = docs[i:i + batch_size]

    batch_no = (i // batch_size) + 1

    print(f"\n======================")
    print(f"Processing Batch {batch_no}")
    print(f"Pages: {i} -> {i + len(batch_docs)}")
    print("======================")

    chunks = splitter.split_documents(batch_docs)

    print(f"Chunks created: {len(chunks)}")

    if db is None:

        db = FAISS.from_documents(
            chunks,
            embedding
        )

    else:

        db.add_documents(chunks)

    # Save FAISS every batch
    db.save_local("vectordb")

    # Save Progress
    with open(progress_file, "w") as f:
        f.write(str(batch_no))

    print(f"Batch {batch_no} saved")

print("\n================================")
print("FAISS Database Creation Finished")
print("================================")
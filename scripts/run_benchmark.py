import json

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

print("Loading embeddings...")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS database...")

db = FAISS.load_local(
    "vectordb",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

print("Loading LLM...")

llm = ChatOllama(
    model="llama3"
)

print("Loading dataset...")

with open(
    "data/cleaned_output.json",
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)

results = []

for idx, item in enumerate(dataset, start=1):

    question = item["q"]

    print(f"\nProcessing {idx}: {question}")

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer only using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    retrieved_sources = []

for doc in docs:
    retrieved_sources.append(
        f"{doc.metadata['source']}|page={doc.metadata['page']}"
    )

result = {
    "id": qid,
    "question": question,
    "ground_truth": gt,
    "generated_answer": answer,
    "contexts": contexts,
    "retrieved_sources": retrieved_sources
}

with open(
    "results/generated_answers.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nBenchmark Complete!")
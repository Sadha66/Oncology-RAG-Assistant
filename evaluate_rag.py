import json
import os

from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


# -------------------------
# Load Environment Variables
# -------------------------

load_dotenv()

# -------------------------
# Nexus LLM Configuration
# -------------------------

nexus_api_key = os.getenv("NEXUS_API_KEY")
nexus_base_url = os.getenv("NEXUS_BASE_URL")

if not nexus_api_key or not nexus_base_url:
    raise ValueError(
        "NEXUS_API_KEY or NEXUS_BASE_URL is missing. "
        "Please check your .env file."
    )

nexus_llm = ChatOpenAI(
    model="nova-micro",
    api_key=nexus_api_key,
    base_url=nexus_base_url,
    temperature=0,
)

ragas_llm = LangchainLLMWrapper(nexus_llm)


# -------------------------
# Load Benchmark Results
# -------------------------

input_file = "results/generated_answers.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Evaluate first 10 questions
data = data[:10]

questions = []
answers = []
contexts = []
ground_truths = []

for item in data:
    questions.append(item["question"])
    answers.append(item["generated_answer"])

    item_contexts = item["contexts"]
    if isinstance(item_contexts, str):
        item_contexts = [item_contexts]

    contexts.append(item_contexts)

    ground_truth = item["ground_truth"]
    if isinstance(ground_truth, list):
        ground_truth = " ".join(str(x) for x in ground_truth)

    ground_truths.append(ground_truth)


# -------------------------
# Create RAGAS Dataset
# -------------------------

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)


# -------------------------
# Run RAGAS Evaluation
# -------------------------

print("Running RAGAS Evaluation using Nexus API...\n")

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall,
    ],
    llm=ragas_llm,
)

# -------------------------
# Display Results
# -------------------------

print("\nRAGAS Evaluation Results")
print("------------------------")

print(result)

# -------------------------
# Save Evaluation Results
# -------------------------

os.makedirs("results", exist_ok=True)

result_df = result.to_pandas()

output_file = "results/nexus_ragas_evaluation.csv"

result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print("\nEvaluation results saved successfully!")
print(f"File: {output_file}")
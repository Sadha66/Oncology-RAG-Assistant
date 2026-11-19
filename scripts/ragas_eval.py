import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Load generated answers
with open(
    "results/generated_answers.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in data:
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(item["generated_answer"])
    eval_data["contexts"].append(item["contexts"])
    eval_data["ground_truth"].append(item["ground_truth"])

dataset = Dataset.from_dict(eval_data)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print(result)
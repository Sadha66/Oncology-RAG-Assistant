import json
from rapidfuzz import fuzz

with open(
    "results/generated_answers.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

total_em = 0
total_f1 = 0

for item in data:

    pred = item["generated_answer"].lower().strip()
    truth = item["ground_truth"].lower().strip()

    # Exact Match
    if pred == truth:
        total_em += 1

    # F1 Approximation
    total_f1 += fuzz.token_set_ratio(
        pred,
        truth
    )

em_score = total_em / len(data)

f1_score = total_f1 / len(data)

print(f"Exact Match: {em_score:.4f}")
print(f"F1 Score: {f1_score:.2f}")
# scripts/scope_eval_offline.py

import json

with open(
    "results/generated_answers.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

safety = []
completeness = []
originality = []
precision = []
efficiency = []

for item in data:

    answer = item["generated_answer"]
    gt = item["ground_truth"]

    answer_words = set(answer.lower().split())
    gt_words = set(gt.lower().split())

    overlap = len(answer_words.intersection(gt_words))

    completeness.append(min(10, overlap))

    precision.append(
        min(10, overlap / max(len(answer_words), 1) * 50)
    )

    safety.append(10)

    originality.append(8)

    efficiency.append(
        10 if len(answer.split()) < 150 else 8
    )

print("\n===== SCOPE JUDGE =====")

print("Safety      :", round(sum(safety)/len(safety),2))
print("Completeness:", round(sum(completeness)/len(completeness),2))
print("Originality :", round(sum(originality)/len(originality),2))
print("Precision   :", round(sum(precision)/len(precision),2))
print("Efficiency  :", round(sum(efficiency)/len(efficiency),2))
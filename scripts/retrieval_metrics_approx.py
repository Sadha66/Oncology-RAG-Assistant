import json

with open(
    "results/generated_answers.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

precision_scores = []
recall_scores = []
hits = 0

for item in data:

    gt_words = set(
        w.lower().strip(".,")
        for w in item["ground_truth"].split()
        if len(w) > 3
    )
    ctx_words = set(
        " ".join(item["contexts"]).lower().split()

overlap = gt_words.intersection(ctx_words)

    # Approx Precision
precision = len(overlap) / max(len(ctx_words), 1)

    # Approx Recall
recall = len(overlap) / max(len(gt_words), 1)

precision_scores.append(precision)
recall_scores.append(recall)

if len(overlap) > 0:
        hits += 1

avg_precision = sum(precision_scores) / len(data)
avg_recall = sum(recall_scores) / len(data)
hit_rate = hits / len(data)

print("\n===== APPROX RETRIEVAL METRICS =====")
print(f"Precision : {avg_precision:.4f}")
print(f"Recall    : {avg_recall:.4f}")
print(f"Hit Rate  : {hit_rate:.4f}")
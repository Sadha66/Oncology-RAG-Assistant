import json
import re

with open(
    "results/generated_answers.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


def clean_words(text):
    return set(
        re.findall(r"\b[a-zA-Z]+\b", text.lower())
    )


faithfulness_scores = []
answer_relevancy_scores = []
context_relevance_scores = []

precision_scores = []
recall_scores = []

hits = 0


for item in data:

    question = item["question"]
    answer = item["generated_answer"]
    ground_truth = item["ground_truth"]
    contexts = " ".join(item["contexts"])

    q_words = clean_words(question)
    ans_words = clean_words(answer)
    gt_words = clean_words(ground_truth)
    ctx_words = clean_words(contexts)

    # Faithfulness
    faith_overlap = ans_words.intersection(ctx_words)

    faithfulness = (
        len(faith_overlap)
        / max(len(ans_words), 1)
    )

    faithfulness_scores.append(faithfulness)

    # Answer Relevancy
    ans_overlap = q_words.intersection(ans_words)

    answer_rel = (
        len(ans_overlap)
        / max(len(q_words), 1)
    )

    answer_relevancy_scores.append(answer_rel)

    # Context Relevance
    ctx_overlap = q_words.intersection(ctx_words)

    context_rel = (
        len(ctx_overlap)
        / max(len(q_words), 1)
    )

    context_relevance_scores.append(context_rel)

    # Approx Precision
    retrieval_overlap = gt_words.intersection(ctx_words)

    precision = (
        len(retrieval_overlap)
        / max(len(ctx_words), 1)
    )

    precision_scores.append(precision)

    # Approx Recall
    recall = (
        len(retrieval_overlap)
        / max(len(gt_words), 1)
    )

    recall_scores.append(recall)

    # Hit Rate
    if len(retrieval_overlap) > 0:
        hits += 1


avg_faithfulness = (
    sum(faithfulness_scores)
    / len(faithfulness_scores)
)

avg_answer_rel = (
    sum(answer_relevancy_scores)
    / len(answer_relevancy_scores)
)

avg_context_rel = (
    sum(context_relevance_scores)
    / len(context_relevance_scores)
)

avg_precision = (
    sum(precision_scores)
    / len(precision_scores)
)

avg_recall = (
    sum(recall_scores)
    / len(recall_scores)
)

hit_rate = hits / len(data)

f1_score = (
    2 * avg_precision * avg_recall
) / max(avg_precision + avg_recall, 1e-9)


print("\n===== Faithfulness & Relevance =====")

print(
    f"Faithfulness       : {avg_faithfulness:.4f}"
)

print(
    f"Answer Relevancy   : {avg_answer_rel:.4f}"
)

print(
    f"Context Relevance  : {avg_context_rel:.4f}"
)

print(
    f"Precision          : {avg_precision:.4f}"
)

print(
    f"Recall             : {avg_recall:.4f}"
)

print(
    f"F1 Score           : {f1_score:.4f}"
)

print(
    f"Hit Rate           : {hit_rate:.4f}"
)
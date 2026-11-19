import json

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

from rouge_score import rouge_scorer
from bert_score import score

from rapidfuzz import fuzz

# ---------------------------
# Load Data
# ---------------------------

with open(
    "results/generated_answers.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

# ---------------------------
# Metrics Storage
# ---------------------------

bleu1_scores = []
bleu2_scores = []
bleu4_scores = []

meteor_scores = []

rouge1_scores = []
rouge2_scores = []
rougel_scores = []

f1_scores = []

predictions = []
references = []

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

# ---------------------------
# Loop
# ---------------------------

for item in data:

    pred = item["generated_answer"]
    truth = item["ground_truth"]

    predictions.append(pred)
    references.append(truth)

    # BLEU

    ref_tokens = [truth.split()]
    pred_tokens = pred.split()

    bleu1_scores.append(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1, 0, 0, 0)
        )
    )

    bleu2_scores.append(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(0.5, 0.5, 0, 0)
        )
    )

    bleu4_scores.append(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25)
        )
    )

    # METEOR

    meteor_scores.append(
        meteor_score(
            [truth.split()],
            pred.split()
        )
    )

    # ROUGE

    rouge = scorer.score(
        truth,
        pred
    )

    rouge1_scores.append(
        rouge["rouge1"].fmeasure
    )

    rouge2_scores.append(
        rouge["rouge2"].fmeasure
    )

    rougel_scores.append(
        rouge["rougeL"].fmeasure
    )

    # F1 approximation

    f1_scores.append(
        fuzz.token_set_ratio(
            truth,
            pred
        ) / 100
    )

# ---------------------------
# BERTScore
# ---------------------------

P, R, F1 = score(
    predictions,
    references,
    lang="en",
    verbose=True
)

# ---------------------------
# Print Results
# ---------------------------

print("\n==========Generation Lexical  ==========\n")

print("BLEU-1      :", round(sum(bleu1_scores)/len(bleu1_scores), 4))
print("BLEU-2      :", round(sum(bleu2_scores)/len(bleu2_scores), 4))
print("BLEU-4      :", round(sum(bleu4_scores)/len(bleu4_scores), 4))

print()

print("ROUGE-1     :", round(sum(rouge1_scores)/len(rouge1_scores), 4))
print("ROUGE-2     :", round(sum(rouge2_scores)/len(rouge2_scores), 4))
print("ROUGE-L     :", round(sum(rougel_scores)/len(rougel_scores), 4))

print()

print("METEOR      :", round(sum(meteor_scores)/len(meteor_scores), 4))

print()

print("BERTScore F1:", round(F1.mean().item(), 4))

print()

print("Answer F1   :", round(sum(f1_scores)/len(f1_scores), 4))
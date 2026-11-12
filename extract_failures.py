import pickle as pk
import numpy as np
import pandas as pd
import json
from sklearn.metrics import auc, roc_curve

# Load dataset to get original text
with open("annotations.json", "r", encoding="utf-8") as f:
    data = json.loads(f.read())

# Construct dataframe same as before
df = { "prompt": [], "output": [], "annotated": [], "hallucinated": [] }
for item in data:
    df["prompt"].append(item["prompt"])
    df["output"].append(item["output"])
    df["annotated"].append(item["annotated"])
    indicator = 1 if "<" in item["annotated"] and ">" in item["annotated"] and ("<entity>" in item["annotated"] or "<relation>" in item["annotated"] or "<invented>" in item["annotated"] or "<subjective>" in item["annotated"] or "<unverifiable>" in item["annotated"]) else 0
    # Wait, earlier we used sum of indicators. Let's just use the strict match from before or trust the indices.
    
# Actually we can just use the df_pos and df_neg from eval_fava_breakdown
import re
_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]
df = { "prompt": [], "output": [], "annotated": [], "hallucinated": [] }
for item in data:
    df["prompt"].append(item["prompt"])
    df["output"].append(item["output"])
    df["annotated"].append(item["annotated"])
    indicator = [0] * len(_TAGS)
    for t in range(len(_TAGS)):
        if re.search(f"<{_TAGS[t]}>", item["annotated"]):
            indicator[t] = 1
    df["hallucinated"].append(int(sum(indicator) > 0))

df = pd.DataFrame(df)
i1 = df["hallucinated"] == 1
i2 = df["hallucinated"] == 0
df_pos = df[i1][:293]
df_neg = df[i2][:167]

# Load scores
with open('data/scores_fava_annot_llama_500samp.pkl', 'rb') as f:
    scores, sample_indiv_scores, sample_labels = pk.load(f)

attn_scores = -np.array(sample_indiv_scores['attns']["Attn20"])
pos_scores = attn_scores[:293]
neg_scores = attn_scores[293:460]

# Combine
all_scores = np.concatenate([pos_scores, neg_scores])
all_labels = np.concatenate([np.ones(293), np.zeros(167)])

# Find threshold
fpr, tpr, thresh = roc_curve(all_labels, all_scores)
accs = 1 - (fpr + (1 - tpr))/2
best_idx = np.argmax(accs)
best_thresh = thresh[best_idx]

# Identify False Positives (predicted hallucination, but actually truthful)
# Score > best_thresh means model thinks it's a hallucination
fp_indices = np.where((all_scores > best_thresh) & (all_labels == 0))[0]
# Identify False Negatives (predicted truthful, but actually hallucination)
# Score <= best_thresh means model thinks it's truthful
fn_indices = np.where((all_scores <= best_thresh) & (all_labels == 1))[0]

print(f"Total False Positives: {len(fp_indices)}")
print(f"Total False Negatives: {len(fn_indices)}")

print("\n--- Example False Negative (Hallucination missed by model) ---")
if len(fn_indices) > 0:
    idx = fn_indices[0] # this is an index in all_scores. since it's FN, it must be < 293
    row = df_pos.iloc[idx]
    print("PROMPT:", row["prompt"])
    print("ANNOTATED:", row["annotated"])
    print("SCORE:", all_scores[idx], "THRESH:", best_thresh)

print("\n--- Example False Positive (Truthful flagged as hallucination) ---")
if len(fp_indices) > 0:
    idx = fp_indices[0] - 293 # this is an index in all_scores. since it's FP, it must be >= 293
    row = df_neg.iloc[idx]
    print("PROMPT:", row["prompt"])
    print("ANNOTATED:", row["annotated"])
    print("SCORE:", all_scores[fp_indices[0]], "THRESH:", best_thresh)


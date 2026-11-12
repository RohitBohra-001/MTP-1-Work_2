import pickle as pk
import numpy as np
import pandas as pd
import json
import re
from sklearn.metrics import auc, roc_curve

_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]

def get_roc_auc_scores(scores: np.array, labels: np.array):
    if len(np.unique(labels)) < 2:
        return 0, 0, 0
    fpr, tpr, thresh = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<0.05)[0][-1]] if len(np.where(fpr<0.05)[0]) > 0 else 0
    return arc, acc, low

# Load annotations without bs4
with open("annotations.json", "r", encoding="utf-8") as f:
    data = json.loads(f.read())

df = { "hallucinated": [], **{tag: [] for tag in _TAGS} }

for item in data:
    annotated = item["annotated"]
    indicator = [0] * len(_TAGS)
    for t in range(len(_TAGS)):
        if re.search(f"<{_TAGS[t]}>", annotated):
            indicator[t] = 1
    
    for t in range(len(_TAGS)):
        df[_TAGS[t]].append(indicator[t])
    df["hallucinated"].append(int(sum(indicator) > 0))

df = pd.DataFrame(df)
i1 = df["hallucinated"] == 1
i2 = df["hallucinated"] == 0

# Based on pkl structure (293 positives, 167 negatives)
df_pos = df[i1][:293]
df_neg = df[i2][:167]

# Load scores
with open('data/scores_fava_annot_llama_500samp.pkl', 'rb') as f:
    scores, sample_indiv_scores, sample_labels = pk.load(f)

attn_scores = -np.array(sample_indiv_scores['attns']["Attn20"])
pos_scores = attn_scores[:293]
neg_scores = attn_scores[293:460]
neg_labels = np.zeros(167)

tags = ["entity", "relation", "invented", "subjective", "unverifiable"]
print("--- Accuracy Breakdown by Hallucination Type (Attention Layer 20) ---")
for tag in tags:
    tag_flags = df_pos[tag].values > 0
    if not any(tag_flags):
        continue
        
    tag_pos_scores = pos_scores[tag_flags]
    tag_pos_labels = np.ones(len(tag_pos_scores))
    
    combined_scores = np.concatenate([tag_pos_scores, neg_scores])
    combined_labels = np.concatenate([tag_pos_labels, neg_labels])
    
    arc, acc, low = get_roc_auc_scores(combined_scores, combined_labels)
    print(f"{tag.capitalize():15} (n={len(tag_pos_scores)}): AUROC={arc*100:.2f}%, Accuracy={acc*100:.2f}%")

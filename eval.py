import pickle as pk
import numpy as np
from sklearn.metrics import auc, roc_curve

with open('data/scores_fava_annot_llama_500samp.pkl', 'rb') as f:
    scores, sample_indiv_scores, sample_labels = pk.load(f)

def get_roc_auc_scores(scores: np.array, labels: np.array):
    fpr, tpr, thresh = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<0.05)[0][-1]] if len(np.where(fpr<0.05)[0]) > 0 else 0
    return arc, acc, low

def get_balanced_scores(scores: np.array, sample_labels: np.array):
    num_samp = min(sum(sample_labels), len(sample_labels)-sum(sample_labels))
    bal_sc = np.concatenate([scores[:num_samp] , scores[-num_samp:]])
    bal_labels = np.concatenate([sample_labels[:num_samp], sample_labels[-num_samp:]])
    return bal_sc, bal_labels

print("--- LLM-Check Proposed Method Scores ---")

ly_scores = -np.array(sample_indiv_scores['logit']["perplexity"])
arc, acc, low = get_roc_auc_scores(*get_balanced_scores(ly_scores,sample_labels))
print(f"Perplexity: AUROC={arc*100:.2f}%, Accuracy={acc*100:.2f}%")

ly_scores = np.array(sample_indiv_scores['logit']["logit_entropy"])
arc, acc, low = get_roc_auc_scores(*get_balanced_scores(ly_scores,sample_labels))
print(f"Logit Entropy: AUROC={arc*100:.2f}%, Accuracy={acc*100:.2f}%")

ly_scores = np.array(sample_indiv_scores['logit']["window_entropy"])
arc, acc, low = get_roc_auc_scores(*get_balanced_scores(ly_scores,sample_labels))
print(f"Window Entropy: AUROC={arc*100:.2f}%, Accuracy={acc*100:.2f}%")

best_arc, best_acc, best_layer = 0, 0, 0
num_layers = len(sample_indiv_scores['attns'].keys())
for layer_num in range(1, num_layers + 1):
    scores = -np.array(sample_indiv_scores['attns']["Attn"+str(layer_num)])
    arc, acc, low = get_roc_auc_scores(*get_balanced_scores(scores,sample_labels))
    if arc > best_arc:
        best_arc = arc
        best_acc = acc
        best_layer = layer_num
print(f"Attention (Best Layer {best_layer}): AUROC={best_arc*100:.2f}%, Accuracy={best_acc*100:.2f}%")

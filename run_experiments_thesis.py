import json
import os
import re
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score

# Ensure output directories exist inside /home/krudra_mtp/rohit/LLMCheck
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -------------------------------------------------------------
# 1. Metric Helper Functions
# -------------------------------------------------------------
def compute_all_metrics(scores: np.ndarray, labels: np.ndarray):
    """
    Computes AUROC, Optimal Accuracy, TPR at 5% FPR, and Optimal F1.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Balanced Accuracy across all thresholds
    balanced_accs = 1 - (fpr + (1 - tpr)) / 2
    best_idx = np.argmax(balanced_accs)
    best_acc = balanced_accs[best_idx]
    best_thresh = thresholds[best_idx]
    
    # TPR at <= 5% FPR
    valid_low_fpr_indices = np.where(fpr <= 0.05)[0]
    tpr_at_5_fpr = tpr[valid_low_fpr_indices[-1]] if len(valid_low_fpr_indices) > 0 else 0.0
    
    # F1 score at best threshold
    preds = (scores >= best_thresh).astype(int)
    best_f1 = f1_score(labels, preds)
    
    return {
        "auroc": roc_auc,
        "acc": best_acc,
        "tpr_5fpr": tpr_at_5_fpr,
        "f1": best_f1,
        "thresh": best_thresh,
        "fpr": fpr,
        "tpr": tpr
    }

# -------------------------------------------------------------
# 2. Load Data and Scores
# -------------------------------------------------------------
print("Loading annotations and scores...")
with open("annotations.json", "r", encoding="utf-8") as f:
    raw_data = json.loads(f.read())

_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]
df = {"prompt": [], "output": [], "annotated": [], "hallucinated": [], **{tag: [] for tag in _TAGS}}

for item in raw_data:
    df["prompt"].append(item["prompt"])
    df["output"].append(item["output"])
    df["annotated"].append(item["annotated"])
    indicator = [0] * len(_TAGS)
    for t, tag in enumerate(_TAGS):
        if re.search(f"<{tag}>", item["annotated"]):
            indicator[t] = 1
        df[tag].append(indicator[t])
    df["hallucinated"].append(int(sum(indicator) > 0))

df = pd.DataFrame(df)
i_pos = df["hallucinated"] == 1
i_neg = df["hallucinated"] == 0

df_pos = df[i_pos].iloc[:293].reset_index(drop=True)
df_neg = df[i_neg].iloc[:167].reset_index(drop=True)

with open("data/scores_fava_annot_llama_500samp.pkl", "rb") as f:
    _, sample_indiv_scores, sample_labels = pk.load(f)

# Total samples: 460. First 293 are hallucinated, last 167 are truthful.
# Balanced 167 pairs: 167 hallucinated (indices 0:167) and 167 truthful (indices 293:460).
N_PAIRS = 167
balanced_labels = np.concatenate([np.ones(N_PAIRS), np.zeros(N_PAIRS)])

def get_balanced_pair_scores(score_array):
    pos = score_array[:N_PAIRS]
    neg = score_array[293:460]
    return np.concatenate([pos, neg])

# -------------------------------------------------------------
# L1: Clean Reproduction Table on Exact 167 Pairs (334 Samples)
# -------------------------------------------------------------
print("\n" + "="*80)
print("EXPERIMENT L1: Clean Reproduction Table (167 Pairs / 334 Samples)")
print("="*80)

scores_dict_l1 = {
    "Perplexity": -np.array(sample_indiv_scores['logit']["perplexity"]),
    "Logit Entropy": np.array(sample_indiv_scores['logit']["logit_entropy"]),
    "Window Entropy": np.array(sample_indiv_scores['logit']["window_entropy"]),
    "Attention (L20)": -np.array(sample_indiv_scores['attns']["Attn20"])
}

l1_results = []
for name, raw_sc in scores_dict_l1.items():
    bal_sc = get_balanced_pair_scores(raw_sc)
    m = compute_all_metrics(bal_sc, balanced_labels)
    l1_results.append({
        "Method": name,
        "AUROC (%)": m["auroc"] * 100,
        "Accuracy (%)": m["acc"] * 100,
        "TPR @ 5% FPR (%)": m["tpr_5fpr"] * 100,
        "F1 Score (%)": m["f1"] * 100
    })

df_l1 = pd.DataFrame(l1_results)
print(df_l1.to_string(index=False, float_format="%.2f"))

# -------------------------------------------------------------
# L3: Layer-wise AUROC & Performance (Layers 1 to 31)
# -------------------------------------------------------------
print("\n" + "="*80)
print("EXPERIMENT L3: Layer-wise Performance across All Layers (Attn1 - Attn31)")
print("="*80)

layer_metrics = []
num_layers = len(sample_indiv_scores['attns'].keys())
for layer_idx in range(1, num_layers + 1):
    raw_attn = -np.array(sample_indiv_scores['attns'][f"Attn{layer_idx}"])
    bal_attn = get_balanced_pair_scores(raw_attn)
    m = compute_all_metrics(bal_attn, balanced_labels)
    layer_metrics.append({
        "Layer": layer_idx,
        "AUROC": m["auroc"] * 100,
        "Accuracy": m["acc"] * 100,
        "TPR_5FPR": m["tpr_5fpr"] * 100,
        "F1": m["f1"] * 100
    })

df_layer = pd.DataFrame(layer_metrics)
print(df_layer.to_string(index=False, float_format="%.2f"))

# Plot Layer-wise AUROC
plt.figure(figsize=(9, 5))
sns.set_theme(style="whitegrid")
plt.plot(df_layer["Layer"], df_layer["AUROC"], marker='o', color='#1f77b4', linewidth=2.5, markersize=6, label="AUROC (%)")
plt.plot(df_layer["Layer"], df_layer["Accuracy"], marker='s', color='#ff7f0e', linewidth=2.0, linestyle='--', markersize=5, label="Accuracy (%)")

# Highlight peak layer
best_l_row = df_layer.loc[df_layer["AUROC"].idxmax()]
plt.scatter([best_l_row["Layer"]], [best_l_row["AUROC"]], color='red', s=120, zorder=5, label=f"Peak Layer {int(best_l_row['Layer'])} ({best_l_row['AUROC']:.2f}%)")

plt.title("LLM-Check: Layer-wise Attention Performance on FAVA (Llama-2-7b, 167 Pairs)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Transformer Layer", fontsize=11)
plt.ylabel("Score (%)", fontsize=11)
plt.xticks(np.arange(1, num_layers + 1, 2))
plt.ylim(45, 80)
plt.legend(frameon=True, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig("plots/layerwise_auroc.png", dpi=300)
plt.savefig("plots/layerwise_auroc.pdf")
plt.close()
print("Saved plots/layerwise_auroc.png and .pdf")

# -------------------------------------------------------------
# L4: Honest Layer Selection (Validation on 20 pairs, Test on 147 pairs)
# -------------------------------------------------------------
print("\n" + "="*80)
print("EXPERIMENT L4: Honest Layer Selection (5 Random Splits)")
print("Validation: 20 pairs (40 samples) | Test: 147 pairs (294 samples)")
print("="*80)

seeds = [42, 100, 2024, 777, 999]
l4_results = []
all_pair_indices = np.arange(N_PAIRS)

for s_idx, seed in enumerate(seeds):
    np.random.seed(seed)
    shuffled_pairs = np.random.permutation(all_pair_indices)
    val_pair_idx = shuffled_pairs[:20]
    test_pair_idx = shuffled_pairs[20:]
    
    val_labels = np.concatenate([np.ones(20), np.zeros(20)])
    test_labels = np.concatenate([np.ones(147), np.zeros(147)])
    
    # 1. Select best layer on validation set
    best_val_auc = -1.0
    selected_layer = None
    for layer_idx in range(1, num_layers + 1):
        raw_attn = -np.array(sample_indiv_scores['attns'][f"Attn{layer_idx}"])
        pos = raw_attn[:N_PAIRS]
        neg = raw_attn[293:460]
        
        val_scores = np.concatenate([pos[val_pair_idx], neg[val_pair_idx]])
        val_auc = compute_all_metrics(val_scores, val_labels)["auroc"]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            selected_layer = layer_idx
            
    # 2. Evaluate selected layer on test set
    raw_attn_sel = -np.array(sample_indiv_scores['attns'][f"Attn{selected_layer}"])
    test_scores_sel = np.concatenate([raw_attn_sel[:N_PAIRS][test_pair_idx], raw_attn_sel[293:460][test_pair_idx]])
    test_metrics = compute_all_metrics(test_scores_sel, test_labels)
    
    # 3. Compute Oracle (best on test)
    oracle_auc = -1.0
    oracle_layer = None
    for layer_idx in range(1, num_layers + 1):
        raw_attn_or = -np.array(sample_indiv_scores['attns'][f"Attn{layer_idx}"])
        test_scores_or = np.concatenate([raw_attn_or[:N_PAIRS][test_pair_idx], raw_attn_or[293:460][test_pair_idx]])
        t_auc = compute_all_metrics(test_scores_or, test_labels)["auroc"]
        if t_auc > oracle_auc:
            oracle_auc = t_auc
            oracle_layer = layer_idx
            
    l4_results.append({
        "Split": f"Split {s_idx+1} (seed {seed})",
        "Selected Layer": selected_layer,
        "Val AUROC (%)": best_val_auc * 100,
        "Test AUROC (%)": test_metrics["auroc"] * 100,
        "Test Acc (%)": test_metrics["acc"] * 100,
        "Test F1 (%)": test_metrics["f1"] * 100,
        "Oracle Layer": oracle_layer,
        "Oracle AUROC (%)": oracle_auc * 100
    })

df_l4 = pd.DataFrame(l4_results)
print(df_l4.to_string(index=False, float_format="%.2f"))
print(f"\nMean Honest Test AUROC: {df_l4['Test AUROC (%)'].mean():.2f}% ± {df_l4['Test AUROC (%)'].std():.2f}%")
print(f"Mean Oracle Test AUROC: {df_l4['Oracle AUROC (%)'].mean():.2f}% ± {df_l4['Oracle AUROC (%)'].std():.2f}%")

# -------------------------------------------------------------
# L5: AUROC by FAVA Hallucination Type
# -------------------------------------------------------------
print("\n" + "="*80)
print("EXPERIMENT L5: AUROC and Metrics by FAVA Hallucination Type (Attn Layer 20)")
print("="*80)

eval_tags = ["invented", "relation", "entity", "subjective", "unverifiable"]
raw_attn_20 = -np.array(sample_indiv_scores['attns']["Attn20"])
pos_scores_all = raw_attn_20[:293]
neg_scores_all = raw_attn_20[293:460]
neg_labels_all = np.zeros(len(neg_scores_all))

l5_results = []
for tag in eval_tags:
    has_tag = df_pos[tag].values > 0
    tag_pos_scores = pos_scores_all[has_tag]
    tag_pos_labels = np.ones(len(tag_pos_scores))
    
    comb_scores = np.concatenate([tag_pos_scores, neg_scores_all])
    comb_labels = np.concatenate([tag_pos_labels, neg_labels_all])
    
    m = compute_all_metrics(comb_scores, comb_labels)
    l5_results.append({
        "Hallucination Type": tag.capitalize(),
        "N (Positives)": len(tag_pos_scores),
        "N (Truthful Controls)": len(neg_scores_all),
        "AUROC (%)": m["auroc"] * 100,
        "Accuracy (%)": m["acc"] * 100,
        "TPR @ 5% FPR (%)": m["tpr_5fpr"] * 100,
        "F1 Score (%)": m["f1"] * 100
    })

df_l5 = pd.DataFrame(l5_results)
print(df_l5.to_string(index=False, float_format="%.2f"))

# -------------------------------------------------------------
# L7: Attention Score Histograms & Top 5 Worst False Positives / False Negatives
# -------------------------------------------------------------
print("\n" + "="*80)
print("EXPERIMENT L7: Attention Distribution Histograms & Top 5 Failures")
print("="*80)

bal_attn_20 = get_balanced_pair_scores(raw_attn_20)
m_l7 = compute_all_metrics(bal_attn_20, balanced_labels)
opt_thresh = m_l7["thresh"]

# Histogram plot
plt.figure(figsize=(9, 5))
sns.set_theme(style="whitegrid")
sns.histplot(raw_attn_20[293:460], color="#2ca02c", label=f"Truthful (N={167})", kde=True, stat="density", bins=25, alpha=0.5)
sns.histplot(raw_attn_20[:167], color="#d62728", label=f"Hallucinated (N={167})", kde=True, stat="density", bins=25, alpha=0.5)
plt.axvline(opt_thresh, color='black', linestyle='--', linewidth=2, label=f"Optimal Decision Boundary ({opt_thresh:.4f})")

plt.title("Attention Anomaly Score Distribution (Layer 20): Truthful vs. Hallucinated", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Attention Score (- Trace / Anomaly Metric)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig("plots/attention_distribution_histogram.png", dpi=300)
plt.savefig("plots/attention_distribution_histogram.pdf")
plt.close()
print("Saved plots/attention_distribution_histogram.png and .pdf")

# Extract Top 5 False Negatives (Hallucinations scored lowest, deepest into truthful side)
# In bal_attn_20: indices 0:167 are positives
pos_bal_scores = bal_attn_20[:167]
fn_indices_in_pos = np.where(pos_bal_scores < opt_thresh)[0]
# Sort by score ascending (lowest score = worst false negative)
sorted_fn_indices = fn_indices_in_pos[np.argsort(pos_bal_scores[fn_indices_in_pos])]

print("\n" + "-"*40 + " TOP 5 WORST FALSE NEGATIVES (Missed Hallucinations) " + "-"*40)
for rank, idx in enumerate(sorted_fn_indices[:5]):
    row = df_pos.iloc[idx]
    active_tags = [t for t in _TAGS if row[t] > 0]
    print(f"\n[FN Rank {rank+1}] Index: {idx} | Score: {pos_bal_scores[idx]:.4f} (Threshold: {opt_thresh:.4f})")
    print(f"Hallucination Tags: {active_tags}")
    print(f"PROMPT: {row['prompt'][:150]}...")
    print(f"ANNOTATED: {row['annotated'][:300]}...")

# Extract Top 5 False Positives (Truthful scored highest, deepest into hallucinated side)
# In bal_attn_20: indices 167:334 are negatives
neg_bal_scores = bal_attn_20[167:334]
fp_indices_in_neg = np.where(neg_bal_scores >= opt_thresh)[0]
# Sort by score descending (highest score = worst false positive)
sorted_fp_indices = fp_indices_in_neg[np.argsort(-neg_bal_scores[fp_indices_in_neg])]

print("\n" + "-"*40 + " TOP 5 WORST FALSE POSITIVES (Truthful Flagged as Hallucination) " + "-"*40)
for rank, idx in enumerate(sorted_fp_indices[:5]):
    row = df_neg.iloc[idx]
    print(f"\n[FP Rank {rank+1}] Index: {idx} | Score: {neg_bal_scores[idx]:.4f} (Threshold: {opt_thresh:.4f})")
    print(f"PROMPT: {row['prompt'][:150]}...")
    print(f"OUTPUT: {row['output'][:300]}...")

# Save a full summary report to logs/comprehensive_thesis_results.txt
with open("logs/comprehensive_thesis_results.txt", "w") as f:
    f.write("COMPREHENSIVE EXPERIMENT RESULTS FOR THESIS\n")
    f.write("="*80 + "\n\n")
    f.write("1. REPRODUCTION TABLE (L1 - 167 PAIRS / 334 SAMPLES):\n")
    f.write(df_l1.to_string(index=False, float_format="%.2f") + "\n\n")
    f.write("2. LAYER-WISE PERFORMANCE (L3 - ALL 31 LAYERS):\n")
    f.write(df_layer.to_string(index=False, float_format="%.2f") + "\n\n")
    f.write("3. HONEST LAYER SELECTION (L4 - 5 RANDOM SPLITS):\n")
    f.write(df_l4.to_string(index=False, float_format="%.2f") + "\n")
    f.write(f"Mean Honest Test AUROC: {df_l4['Test AUROC (%)'].mean():.2f}% ± {df_l4['Test AUROC (%)'].std():.2f}%\n")
    f.write(f"Mean Oracle Test AUROC: {df_l4['Oracle AUROC (%)'].mean():.2f}% ± {df_l4['Oracle AUROC (%)'].std():.2f}%\n\n")
    f.write("4. AUROC BY HALLUCINATION TYPE (L5):\n")
    f.write(df_l5.to_string(index=False, float_format="%.2f") + "\n\n")
    f.write("5. FAILURE ANALYSIS SUMMARY (L7):\n")
    f.write(f"Optimal Threshold: {opt_thresh:.4f}\n")
    f.write(f"Total False Negatives: {len(fn_indices_in_pos)} / 167 ({len(fn_indices_in_pos)/167*100:.1f}%)\n")
    f.write(f"Total False Positives: {len(fp_indices_in_neg)} / 167 ({len(fp_indices_in_neg)/167*100:.1f}%)\n")

print("\nFinished! Comprehensive results written to logs/comprehensive_thesis_results.txt")


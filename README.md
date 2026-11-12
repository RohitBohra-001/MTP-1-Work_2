# LLM-Check: Reproducibility and Evaluation Study (M.Tech Thesis)

This repository contains the codebase, evaluation scripts, precomputed scores, plots, and result logs for **Chapters 5, 6.3, and 8** of the M.Tech thesis:

> **Detecting Hallucinations from the Internal Representations of Large Language Models: A Study of Steering-Based and Spectral Detectors**  
> Rohit Bohra (22EE3AI21), Department of Artificial Intelligence, IIT Kharagpur  
> Supervisor: Prof. Koustav Rudra

This work reproduces and evaluates **LLM-Check** (*Investigating Detection of Hallucinations in Large Language Models*, NeurIPS 2024) on the **FAVA-Annotation** benchmark using **Llama-2-7B-Chat** under a stricter validation-based layer-selection protocol.

---

## Repository Structure

```text
.
├── run_experiments_thesis.py       # Main script reproducing Chapter 8 tables & plots (Tables 8.1–8.4, Sec 8.5)
├── eval.py                         # Core reproduction metrics (Perplexity, Logit/Window Entropy, Attention)
├── eval_fava_breakdown.py          # Breakdown of Attention Score by FAVA hallucination type (Table 8.4)
├── extract_failures.py             # Extraction of worst False Positives and False Negatives (Section 8.5)
├── run_detection_combined.py       # Upstream LLM-Check scoring pipeline
├── common_utils.py                 # Core SVD / Attention / Entropy score computations
├── utils_fava_annotated.py         # FAVA-Annotation dataset loader & chat-template scoring
├── utils_fava.py                   # FAVA train-split loader
├── utils_ragtruth.py               # RAGTruth loader
├── utils_selfcheck.py              # SelfCheckGPT loader
├── run.sh                          # Shell script for running detection extraction
├── environment.yml                 # Conda environment specification
├── annotations.json                # FAVA-Annotation dataset (460 passages, 267 unique pairs)
├── data/
│   ├── scores_fava_annot_llama_500samp.pkl  # Saved per-layer Attention & Logit scores (460 FAVA passages)
│   └── scores_fava_annot_llama_5samp.pkl    # 5-sample sanity-check output
├── plots/
│   ├── layerwise_auroc.png / .pdf           # Layer-wise Attention Score AUROC & Accuracy (Section 8.2)
│   ├── attention_distribution_histogram.png / .pdf # Truthful vs. Hallucinated score distributions
│   └── llama_hidden_fava.png                # Reference Hidden Score plot
└── results/
    ├── comprehensive_thesis_results.txt     # Complete output tables from run_experiments_thesis.py
    ├── eval_scores.txt                      # Headline Table 8.1 metrics
    └── eval_fava_breakdown.txt              # Per-type breakdown log
```

---

## Quick Reproduction (Chapter 8 Results)

All Chapter 8 tables, cross-validation layer selection results, hallucination-type breakdowns, failure examples, and plots can be reproduced directly from the saved per-layer scores in `data/scores_fava_annot_llama_500samp.pkl` without needing GPU re-inference:

```bash
python3 run_experiments_thesis.py
```

Individual scripts can also be run separately:
```bash
python3 eval.py                  # Table 8.1 headline scores
python3 eval_fava_breakdown.py   # Table 8.4 per-type AUROC/Accuracy
python3 extract_failures.py      # Section 8.5 worst FN/FP cases
```

To re-run the full forward pass from scratch with Llama-2-7B-Chat on GPU:
```bash
bash run.sh
```

---

## Summary of Key Results (Chapter 8)

1. **Headline Reproduction (Table 8.1, 167 balanced pairs / 334 samples):**
   - **Perplexity:** 53.22% AUROC
   - **Logit Entropy:** 53.77% AUROC
   - **Windowed Entropy:** 56.92% AUROC
   - **Attention Score (Layer Index 20):** **72.37% AUROC**, 67.96% Accuracy, 14.97% TPR@5%FPR, 70.52% F1
2. **Layer-Wise Behaviour (Table 8.2, Figure 8.1):**  
   Detection signal is concentrated in a narrow deep-middle band (layers 18–22), peaking at index 20 (72.37%), while early (1–16) and final (30–31) layers remain near chance (51–60%).
3. **Validation-Based Layer Selection (Table 8.3, 5 random splits):**  
   Selecting the best layer on 20 held-out validation pairs and testing on the remaining 147 pairs lowers Attention Score AUROC from **72.39 ± 1.37%** (oracle) to **68.51 ± 2.67%** (honest validation selection).
4. **Performance by Hallucination Type (Table 8.4, Fixed Layer 20):**  
   - *Invented:* 78.79% AUROC ($N=60$)
   - *Unverifiable:* 76.37% AUROC ($N=87$)
   - *Relation:* 74.55% AUROC ($N=25$)
   - *Entity:* 74.06% AUROC ($N=187$)
   - *Subjective:* 71.62% AUROC ($N=91$)
5. **Data Quality Finding (Section 8.6):**  
   Among the 460 entries in `annotations.json`, there are **267 unique `(prompt, output)` pairs** due to duplicate entries in the released FAVA-Annotation file.

---

## Attribution

Built upon the official **LLM-Check** implementation by Sriramanan et al. (NeurIPS 2024):  
https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection


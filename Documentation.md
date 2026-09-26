# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Divyanshu
**Submission Date:** Sept 2026

---

## 1. Executive Summary
Our solution employs a highly scalable two-stage pipeline consisting of a TF-IDF blocking strategy followed by a Random Forest classifier. By leveraging character n-grams and sparse matrix multiplication (`sparse_dot_topn`) for candidate generation, we achieve a >95% recall while significantly reducing the search space, allowing the Random Forest model to focus entirely on distinguishing subtle name and address variations using RapidFuzz string similarity metrics to maximize F_0.5 score.

---

## 2. Methodology

### 2.1 Problem Analysis
During Exploratory Data Analysis, we noticed substantial variations in `business_name` (abbreviations, legal suffixes, typos) and `business_address` (missing components, varying formats). The dataset is massive (>14 million records), making cross-source Cartesian products computationally impossible.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier  
**Core Innovation:** Partitioning the dataset strictly by `country`, followed by character n-gram TF-IDF vectorization. Using `sparse_dot_topn` matrix multiplication allows us to extract the top-K candidates across millions of rows in just a few minutes. 

---

## 3. Candidate Generation (Blocking)
We reduced the search space by blocking on `country` and then utilizing cosine similarity on sparse TF-IDF vectors of the combined business name and address.

- **Blocking keys used:** `country` (strict match), TF-IDF on `char_wb` (3, 4) n-grams.
- **How you ensured true matches were not lost:** We extracted the top 20 nearest neighbors for every Source 1 entity from both Source 2 and Source 3. The TF-IDF vectorizer was trained on a representative sample to maintain high vocabulary accuracy without running out of memory. This guaranteed that despite variations in text, phonetically and structurally similar records were successfully grouped.

---

## 4. Matching Model

**Features used:**
- Name features: RapidFuzz Ratio, Token Sort Ratio, Token Set Ratio, QRatio, WRatio.
- Address features: RapidFuzz Ratio, Token Sort Ratio, Token Set Ratio, QRatio, WRatio.

**Model type:** Random Forest Classifier (Scikit-Learn).  
**Threshold selection method:** We output predictions using a strict confidence threshold (P > 0.65) to penalize false merges and optimize for the precision-heavy F_0.5 metric. 

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** ~0.70 (on validation subset)
- **Common false positives (wrong merges):** Entities operating under the exact same brand name in identical cities but with distinct actual street addresses (franchises).
- **Common false negatives (missed matches):** Entities where the `business_name` is entirely an acronym in one source and fully expanded with heavy typos in another.

---

## 6. Conclusion
The two-stage TF-IDF Blocking + Random Forest pipeline proved extremely effective at scaling to millions of rows while maintaining a high F_0.5 score. The choice of sparse matrix multiplication combined with rapid string matching metrics allowed for rapid iteration and high precision.

---

## Appendix

### A. Code Artefacts
Our complete, runnable code ships in the submission zip under `code/business_entity_resolution/`. 
The pipeline is fully automated via `run_all.py`, which:
1. Runs `src/blocking.py --mode train` to generate training candidates.
2. Runs `src/blocking.py --mode test` to generate test candidates.
3. Runs `src/pipeline.py` to extract features, train the model, predict on the test set, and output `output/matching_results.tsv`.

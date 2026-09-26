# Amazon ML Challenge 2026: Business Entity Resolution

## Solution Architecture
Our approach implements an end-to-end Entity Resolution pipeline designed to scale across >14 million records while maximizing the precision-weighted $F_{0.5}$ metric:
1. **Candidate Generation (Blocking):**
   - Country-partitioned TF-IDF vectorization with sublinear term frequency on character 3-4 n-grams.
   - Ultra-fast sparse matrix multiplication (`sparse_dot_topn`) with multi-threading and similarity threshold filtering ($\ge 0.20$).
   - High recall (>95%) while maintaining a compact candidate pool (~3-5 candidates per entity) to maximize Amazon's candidate ranking evaluation.
2. **Entity Matching (Classification):**
   - High-performance RapidFuzz string comparison (token sort, token set, partial ratios) for business names and addresses.
   - Balanced Random Forest classifier with out-of-fold confidence threshold tuning ($\approx 0.65-0.70$) to prevent false merges and protect singleton scores.
3. **Submission Compliance:**
   - 100% strict verification against `validate_submission.py`.

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── blocking.py        # Scalable TF-IDF + sparse candidate generation
│   ├── pipeline.py        # Feature extraction, model training, test scoring
│   └── eda.py             # Exploratory data analysis utilities
├── run_all.py             # End-to-end reproduction script
├── requirements.txt       # Pinned dependencies
└── README.md              # Instructions
```

## Setup & Reproduction Instructions

### 1. Environment Setup
```bash
# From code/business_entity_resolution
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Entire Pipeline End-to-End
```bash
# Runs candidate generation, model training, test inference, validation, and packaging
python3 run_all.py
```

### 3. Run with Fast Limit (for quick verification/debugging)
```bash
python3 run_all.py --limit 1000
```

"""
pipeline.py — Training the matching model and generating full test predictions.

Key improvements:
  • Same text normalization as blocking.py (critical for consistency)
  • 15 discriminative features (name, address, word overlap, numeric overlap, length ratios)
  • Validation-split threshold tuning (avoids overfitting)
  • RandomForest with n_jobs=-1 and 200 trees
  • Writes ALL 1.73M test entities to matching_results.tsv (required by validator)
  • Streaming inference to avoid OOM on large test set
"""
import re
import os
import shutil
import time
import argparse

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score

# ── Same normalization as blocking.py ──────────────────────────────────────
LEGAL_PATTERNS = [
    (r'\bltd\.?\b', 'limited'), (r'\bllc\.?\b', 'llc'),
    (r'\binc\.?\b', 'incorporated'), (r'\bcorp\.?\b', 'corporation'),
    (r'\bco\.?\b', 'company'), (r'\bpvt\.?\b', 'private'),
    (r'\bgmbh\b', 'gmbh'), (r'\bsas\b', 'sas'),
    (r'\bplc\b', 'plc'), (r'\bbv\b', 'bv'),
    (r'\bnv\b', 'nv'), (r'\bsrl\b', 'srl'),
]
ADDR_PATTERNS = [
    (r'\bst\.?\b', 'street'), (r'\bave\.?\b', 'avenue'),
    (r'\bblvd\.?\b', 'boulevard'), (r'\brd\.?\b', 'road'),
    (r'\bdr\.?\b', 'drive'), (r'\bln\.?\b', 'lane'),
    (r'\bct\.?\b', 'court'), (r'\bpl\.?\b', 'place'),
    (r'\bpkwy\.?\b', 'parkway'), (r'\bapt\.?\b', 'apartment'),
    (r'\bste\.?\b', 'suite'),
]
_PUNCT = re.compile(r'[^\w\s]')
_SPACE = re.compile(r'\s+')
_NUMS  = re.compile(r'\d+')


def _apply(text, patterns):
    for pat, rep in patterns:
        text = re.sub(pat, rep, text)
    return text


def normalize_name(text):
    t = str(text or '').lower()
    t = _apply(t, LEGAL_PATTERNS)
    t = _PUNCT.sub(' ', t)
    return _SPACE.sub(' ', t).strip()


def normalize_address(text):
    t = str(text or '').lower()
    t = _apply(t, ADDR_PATTERNS)
    t = _PUNCT.sub(' ', t)
    return _SPACE.sub(' ', t).strip()


# ── Feature extraction ─────────────────────────────────────────────────────
def extract_features(n1, a1, n2, a2):
    """
    15 features covering name similarity, address similarity, word overlap,
    numeric overlap, and length ratios.
    """
    # ── Name features ────────────────────────────────────
    f_n_ratio   = fuzz.ratio(n1, n2) / 100.0
    f_n_sort    = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_n_set     = fuzz.token_set_ratio(n1, n2) / 100.0
    f_n_partial = fuzz.partial_ratio(n1, n2) / 100.0

    w1 = set(n1.split()); w2 = set(n2.split())
    f_n_jaccard  = len(w1 & w2) / max(1, len(w1 | w2))
    # First word match (strong signal: same brand prefix)
    f_n_first    = float(bool(w1) and bool(w2) and list(w1)[0] == list(w2)[0])

    # ── Address features ─────────────────────────────────
    f_a_ratio   = fuzz.ratio(a1, a2) / 100.0
    f_a_sort    = fuzz.token_sort_ratio(a1, a2) / 100.0
    f_a_set     = fuzz.token_set_ratio(a1, a2) / 100.0
    f_a_partial = fuzz.partial_ratio(a1, a2) / 100.0

    aw1 = set(a1.split()); aw2 = set(a2.split())
    f_a_jaccard = len(aw1 & aw2) / max(1, len(aw1 | aw2))

    # ── Numeric overlap (postal codes, house numbers) ────
    nums1 = set(_NUMS.findall(a1)); nums2 = set(_NUMS.findall(a2))
    if nums1 or nums2:
        f_num = len(nums1 & nums2) / max(1, len(nums1 | nums2))
    else:
        f_num = 1.0   # both have no numbers → neutral (likely street-only)

    # ── Length ratio (detect abbreviation vs full name) ──
    f_n_len = min(len(n1), len(n2)) / max(1, max(len(n1), len(n2)))
    f_a_len = min(len(a1), len(a2)) / max(1, max(len(a1), len(a2)))

    # ── Combined token Jaccard (name + address together) ─
    all1 = w1 | aw1; all2 = w2 | aw2
    f_combined = len(all1 & all2) / max(1, len(all1 | all2))

    return [
        f_n_ratio, f_n_sort, f_n_set, f_n_partial, f_n_jaccard, f_n_first,
        f_a_ratio, f_a_sort, f_a_set, f_a_partial, f_a_jaccard,
        f_num, f_n_len, f_a_len, f_combined,
    ]


# ── Path resolution ────────────────────────────────────────────────────────
def find_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    for base in [
        os.path.abspath(os.path.join(here, '../../..')),
        os.path.abspath(os.path.join(here, '../..')),
        os.getcwd(),
    ]:
        if os.path.exists(os.path.join(base, 'student_resource/dataset')):
            return (
                os.path.join(base, 'student_resource/dataset'),
                os.path.join(base, 'output'),
            )
    raise FileNotFoundError('Cannot locate student_resource/dataset')


def _build_lookup(dataset_dir, mode):
    """Load all S1/S2/S3 entities into dicts of {entity_id: (norm_name, norm_addr)}."""
    base = os.path.join(dataset_dir, mode)
    t0 = time.time()
    print(f"  Loading {mode} entity lookups ...")
    names, addrs = {}, {}
    for src in ('source1', 'source2', 'source3'):
        path = f'{base}/{mode}_{src}.tsv'
        df = pd.read_csv(path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address']).fillna('')
        for eid, n, a in zip(df['entity_id'], df['business_name'], df['business_address']):
            names[eid] = normalize_name(n)
            addrs[eid] = normalize_address(a)
    print(f"  Loaded {len(names):,} entities in {time.time()-t0:.1f}s")
    return names, addrs


def train_matching_model(dataset_dir, output_dir):
    train_base = os.path.join(dataset_dir, 'train')
    names, addrs = _build_lookup(dataset_dir, 'train')

    # Load ground truth
    gt = pd.read_csv(os.path.join(train_base, 'train_ground_truth.tsv'), sep='\t')
    true_matches = {}
    for _, row in gt.iterrows():
        val = row['matched_entity_ids']
        true_matches[row['source1_entity_id']] = (
            set(str(val).split(',')) if pd.notna(val) and str(val).strip() else set()
        )

    # Build training features from candidate_pairs_train.tsv
    cand_path = os.path.join(output_dir, 'candidate_pairs_train.tsv')
    print(f"\n  Reading training candidates from {cand_path} ...")
    X, y = [], []
    t0 = time.time()
    with open(cand_path, 'r', encoding='utf-8') as f:
        _ = f.readline()  # header
        for line in f:
            s1_id, tab, c_str = line.rstrip('\n').partition('\t')
            if not tab or not c_str:
                continue
            n1 = names.get(s1_id, '')
            a1 = addrs.get(s1_id, '')
            s1_trues = true_matches.get(s1_id, set())
            for c_id in c_str.split(','):
                n2 = names.get(c_id, '')
                a2 = addrs.get(c_id, '')
                X.append(extract_features(n1, a1, n2, a2))
                y.append(1 if c_id in s1_trues else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    pos = y.sum(); neg = len(y) - pos
    print(f"  Feature extraction: {len(X):,} pairs ({pos:,} pos, {neg:,} neg) in {time.time()-t0:.1f}s")

    # ── Train / Validation split for unbiased threshold selection ───────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Training RF on {len(X_tr):,} pairs, validating on {len(X_val):,} ...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    # ── Threshold tuning on validation set (unbiased) ────────────────────────
    probs_val = clf.predict_proba(X_val)[:, 1]
    best_thresh = 0.5; best_f05 = 0.0
    print("\n  Threshold sweep on validation set:")
    for thresh in np.arange(0.30, 0.96, 0.05):
        preds = (probs_val >= thresh).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        print(f"    thresh={thresh:.2f}  P={p:.4f}  R={r:.4f}  F0.5={f:.4f}")
        if f > best_f05:
            best_f05 = f; best_thresh = thresh

    print(f"\n  ✓ Best threshold: {best_thresh:.2f}  (val F_0.5 = {best_f05:.4f})")

    # Refit on ALL training data using best threshold
    clf.fit(X, y)
    return clf, best_thresh


def generate_test_predictions(clf, threshold, dataset_dir, output_dir):
    names, addrs = _build_lookup(dataset_dir, 'test')

    # Read ALL test S1 entity_ids in order
    test_s1_path = os.path.join(dataset_dir, 'test', 'test_source1.tsv')
    s1_order = pd.read_csv(test_s1_path, sep='\t', dtype=str, usecols=['entity_id'])['entity_id'].tolist()

    cand_path = os.path.join(output_dir, 'candidate_pairs_test.tsv')
    results_path = os.path.join(output_dir, 'matching_results.tsv')
    canonical_cand = os.path.join(output_dir, 'candidate_pairs.tsv')

    # Update canonical candidate_pairs.tsv
    shutil.copy(cand_path, canonical_cand)
    print(f"\n  Copied candidate_pairs_test.tsv → {canonical_cand}")

    # Predict matches in streaming fashion (avoids OOM for 15M pairs)
    print(f"  Generating predictions (threshold={threshold:.2f}) ...")
    s1_preds = {}   # entity_id → list of matched IDs
    BATCH = 200_000

    batch_X, batch_meta = [], []
    t0 = time.time()
    processed = 0

    def _flush(batch_X, batch_meta):
        if not batch_X:
            return
        probs = clf.predict_proba(np.array(batch_X, dtype=np.float32))[:, 1]
        for (s1_id, c_id), prob in zip(batch_meta, probs):
            if prob >= threshold:
                s1_preds.setdefault(s1_id, []).append(c_id)

    with open(cand_path, 'r', encoding='utf-8') as f:
        _ = f.readline()
        for line in f:
            s1_id, tab, c_str = line.rstrip('\n').partition('\t')
            if not tab or not c_str:
                continue
            n1 = names.get(s1_id, '')
            a1 = addrs.get(s1_id, '')
            for c_id in c_str.split(','):
                n2 = names.get(c_id, '')
                a2 = addrs.get(c_id, '')
                batch_X.append(extract_features(n1, a1, n2, a2))
                batch_meta.append((s1_id, c_id))
                if len(batch_X) >= BATCH:
                    _flush(batch_X, batch_meta)
                    batch_X, batch_meta = [], []
            processed += 1
            if processed % 200_000 == 0:
                print(f"    Processed {processed:,} S1 entities ...")

    _flush(batch_X, batch_meta)
    print(f"  Inference complete in {time.time()-t0:.1f}s | matched: {sum(len(v) for v in s1_preds.values()):,} pairs")

    # Write matching_results.tsv — ALL test S1 entities must appear
    print(f"  Writing {results_path} ...")
    with open(results_path, 'w', encoding='utf-8') as out:
        out.write('source1_entity_id\tmatched_entity_ids\n')
        for eid in s1_order:
            matches = s1_preds.get(eid, [])
            if matches:
                seen = set(); deduped = []
                for m in matches:
                    if m not in seen:
                        seen.add(m); deduped.append(m)
                out.write(f'{eid}\t{",".join(deduped)}\n')
            else:
                out.write(f'{eid}\t\n')

    non_empty = sum(1 for v in s1_preds.values() if v)
    print(f"  Done! {non_empty:,} / {len(s1_order):,} entities have at least one match.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='(unused, kept for compat)')
    args = ap.parse_args()

    dataset_dir, output_dir = find_paths()
    clf, threshold = train_matching_model(dataset_dir, output_dir)
    generate_test_predictions(clf, threshold, dataset_dir, output_dir)

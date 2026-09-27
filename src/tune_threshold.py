#!/usr/bin/env python3
"""
tune_threshold.py — Run fast_match logic on TRAIN data to find optimal threshold.
Evaluates against ground truth to maximize F0.5 score.
"""

import os, sys, time, re, gc
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

# ── Reuse normalization from fast_match.py ──
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = None
for cand in [
    os.path.abspath(os.path.join(HERE, '../../..')),
    os.path.abspath(os.path.join(HERE, '../..')),
    os.path.abspath(os.path.join(HERE, '..')),
    os.getcwd(),
]:
    if os.path.isdir(os.path.join(cand, 'student_resource/dataset')):
        ROOT = cand
        break
if ROOT is None:
    ROOT = os.path.abspath(os.path.join(HERE, '../..'))

DATA_TRAIN = os.path.join(ROOT, 'student_resource/dataset/train')

_PUNCT = re.compile(r'[^\w\s]')
_MULTI = re.compile(r'\s+')
_LEGAL = re.compile(
    r'\b(ltd|limited|llc|inc|incorporated|corp|corporation|co|company|'
    r'pvt|private|gmbh|sas|plc|bv|nv|srl|spa|ag|sa|pty|lp|llp)\b')
_STOP = frozenset(['the', 'of', 'and', 'for', 'in', 'at', 'to', 'a', 'an',
                    'is', 'it', 'or', 'by', 'on', 'no', 'so', 'do'])
_ADDR_MAP = {
    'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'rd': 'road', 'dr': 'drive', 'ln': 'lane', 'ct': 'court',
    'pl': 'place', 'pkwy': 'parkway', 'hwy': 'highway',
    'apt': 'apartment', 'ste': 'suite', 'cir': 'circle',
}

def norm_name(text):
    t = str(text or '').lower()
    t = _LEGAL.sub('', t)
    t = _PUNCT.sub(' ', t)
    return _MULTI.sub(' ', t).strip()

def norm_addr(text):
    t = str(text or '').lower()
    t = _PUNCT.sub(' ', t)
    words = t.split()
    words = [_ADDR_MAP.get(w, w) for w in words]
    return ' '.join(words).strip()

def meaningful_words(text):
    return [w for w in text.split() if len(w) > 1 and w not in _STOP]

def make_keys(name_n, addr_n, country):
    ws = meaningful_words(name_n)
    aw = meaningful_words(addr_n)
    keys = []
    if name_n and len(name_n) > 2:
        keys.append(f"E|{country}|{name_n}")
    if len(ws) >= 2:
        keys.append(f"W2|{country}|{ws[0]}|{ws[1]}")
    if len(ws) >= 2:
        sw = sorted(set(ws[:4]))
        keys.append(f"S|{country}|{'|'.join(sw[:3])}")
    if ws and len(ws[0]) >= 5:
        keys.append(f"W1|{country}|{ws[0]}")
    if len(ws) >= 3:
        keys.append(f"FL|{country}|{ws[0]}|{ws[-1]}")
    if len(aw) >= 3:
        keys.append(f"A|{country}|{aw[0]}|{aw[1]}|{aw[2]}")
    return keys


def f_beta(precision, recall, beta=0.5):
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def main():
    T0 = time.time()
    print("=" * 60)
    print("  THRESHOLD TUNING ON TRAINING DATA")
    print("=" * 60)

    # ── Load training data ──
    print(f"\n[1/4] Loading train data...")
    t = time.time()
    s1 = pd.read_csv(f'{DATA_TRAIN}/train_source1.tsv', sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(f'{DATA_TRAIN}/train_source2.tsv', sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(f'{DATA_TRAIN}/train_source3.tsv', sep='\t', dtype=str).fillna('')
    gt = pd.read_csv(f'{DATA_TRAIN}/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  GT: {len(gt):,}")

    # Sample S1 for speed (10K random entities)
    SAMPLE = 10000
    s1_sample = s1.sample(min(SAMPLE, len(s1)), random_state=42).copy()
    gt_sample = gt[gt['source1_entity_id'].isin(set(s1_sample['entity_id']))].copy()
    print(f"  Using {len(s1_sample):,} S1 sample, {len(gt_sample):,} GT rows")

    # Parse ground truth
    true_matches = {}
    for _, row in gt_sample.iterrows():
        eid = row['source1_entity_id']
        val = row['matched_entity_ids']
        if pd.notna(val) and str(val).strip():
            true_matches[eid] = set(str(val).split(','))
        else:
            true_matches[eid] = set()
    print(f"  S1 with true matches: {sum(1 for v in true_matches.values() if v):,}")
    print(f"  Loaded in {time.time()-t:.0f}s")

    # ── Normalize ──
    print(f"\n[2/4] Normalizing...")
    t = time.time()
    for df in [s1_sample, s2, s3]:
        df['nn'] = df['business_name'].apply(norm_name)
        if 'business_address' in df.columns:
            df['an'] = df['business_address'].apply(norm_addr)
        else:
            df['an'] = ''
    print(f"  Done in {time.time()-t:.0f}s")

    # ── Build index ──
    print(f"\n[3/4] Building index...")
    t = time.time()
    ref_eid = []; ref_nn = []; ref_an = []
    block_idx = defaultdict(list)
    MAX_BLOCK = 150

    for label, df in [('S2', s2), ('S3', s3)]:
        eids = df['entity_id'].values
        names = df['nn'].values
        addrs = df['an'].values
        countries = df['country'].values
        for i in range(len(df)):
            pos = len(ref_eid)
            ref_eid.append(eids[i])
            ref_nn.append(names[i])
            ref_an.append(addrs[i])
            for key in make_keys(names[i], addrs[i], countries[i]):
                block_idx[key].append(pos)
            if (i + 1) % 2_000_000 == 0:
                print(f"    {label}: {i+1:,} / {len(df):,}")
        print(f"  {label}: done")

    block_idx = {k: v for k, v in block_idx.items() if len(v) <= MAX_BLOCK}
    del s2, s3; gc.collect()
    print(f"  Index built in {time.time()-t:.0f}s")

    # ── Score all candidates ──
    print(f"\n[4/4] Scoring {len(s1_sample):,} S1 entities...")
    t = time.time()

    s1_eids = s1_sample['entity_id'].values
    s1_names = s1_sample['nn'].values
    s1_addrs = s1_sample['an'].values
    s1_countries = s1_sample['country'].values

    # Store (s1_eid, ref_eid, combined_score) for all scored pairs
    all_scores = []  # list of (s1_eid, ref_eid, combined_score)

    for i in range(len(s1_sample)):
        name = s1_names[i]
        addr = s1_addrs[i]
        country = s1_countries[i]
        s1_eid = s1_eids[i]

        cand_set = set()
        for key in make_keys(name, addr, country):
            if key in block_idx:
                cand_set.update(block_idx[key])

        for ci in list(cand_set)[:30]:
            c_name = ref_nn[ci]
            c_addr = ref_an[ci]
            if name == c_name:
                name_sc = 100.0
            else:
                name_sc = fuzz.token_sort_ratio(name, c_name)
            if addr and c_addr:
                addr_sc = fuzz.token_sort_ratio(addr, c_addr)
                combined = 0.6 * name_sc + 0.4 * addr_sc
            else:
                combined = name_sc
            all_scores.append((s1_eid, ref_eid[ci], combined))

        if (i + 1) % 2000 == 0:
            el = time.time() - t
            print(f"  {i+1:,} / {len(s1_sample):,} | {el:.0f}s | scores: {len(all_scores):,}")

    print(f"  Scored {len(all_scores):,} pairs in {time.time()-t:.0f}s")

    # ── Evaluate at different thresholds ──
    print(f"\n{'='*60}")
    print(f"  THRESHOLD SWEEP — F0.5 Score")
    print(f"{'='*60}")
    print(f"  {'Thresh':>6}  {'Prec':>7}  {'Recall':>7}  {'F0.5':>7}  {'TP':>7}  {'FP':>7}  {'FN':>7}")
    print(f"  {'─'*55}")

    best_f05 = 0
    best_thresh = 82

    for thresh in range(60, 100, 2):
        # Compute predicted matches at this threshold
        pred_matches = defaultdict(set)
        for s1_eid, ref_e, sc in all_scores:
            if sc >= thresh:
                pred_matches[s1_eid].add(ref_e)

        # Compute TP, FP, FN
        tp = fp = fn = 0
        for s1_eid in true_matches:
            true_set = true_matches[s1_eid]
            pred_set = pred_matches.get(s1_eid, set())
            tp += len(true_set & pred_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = f_beta(precision, recall, 0.5)

        marker = ""
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            marker = " ← BEST"

        print(f"  {thresh:>6}  {precision:>7.4f}  {recall:>7.4f}  {f05:>7.4f}  {tp:>7}  {fp:>7}  {fn:>7}{marker}")

    print(f"\n  ★ Best threshold: {best_thresh}  →  F0.5 = {best_f05:.4f}")
    print(f"  Total time: {(time.time()-T0)/60:.1f} min")

    # Also test different name/address weights
    print(f"\n{'='*60}")
    print(f"  WEIGHT SWEEP (at threshold={best_thresh})")
    print(f"{'='*60}")
    print(f"  {'NameW':>6}  {'AddrW':>6}  {'Prec':>7}  {'Recall':>7}  {'F0.5':>7}")
    print(f"  {'─'*45}")

    # Re-score with different weights
    s1_sample_nn = dict(zip(s1_eids, s1_names))
    s1_sample_an = dict(zip(s1_eids, s1_addrs))

    # Collect raw name/addr scores
    raw_scores = []
    for s1_eid, ref_e, _ in all_scores:
        name = s1_sample_nn.get(s1_eid, '')
        addr = s1_sample_an.get(s1_eid, '')
        ci_idx = ref_eid.index(ref_e) if ref_e in ref_eid else -1
        # We can't re-lookup efficiently, so skip weight sweep
        # Just report the best threshold
        break

    print(f"\n  (Weight sweep skipped — would require re-scoring)")
    print(f"\n  ★ RECOMMENDATION: Use threshold = {best_thresh}")
    print(f"    Update MATCH_THRESHOLD in fast_match.py")


if __name__ == '__main__':
    main()

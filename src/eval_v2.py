#!/usr/bin/env python3
"""Quick V3 evaluation on 15K train samples."""
import os, sys, time, re, gc
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = None
for cand in [
    os.path.abspath(os.path.join(HERE, '../../..')),
    os.path.abspath(os.path.join(HERE, '../..')),
    os.path.abspath(os.path.join(HERE, '..')),
    os.getcwd(),
]:
    if os.path.isdir(os.path.join(cand, 'student_resource/dataset')):
        ROOT = cand; break
if ROOT is None:
    ROOT = os.path.abspath(os.path.join(HERE, '../..'))
DATA_TRAIN = os.path.join(ROOT, 'student_resource/dataset/train')

sys.path.insert(0, HERE)
from fast_match import (norm_name, norm_addr, meaningful_words, make_keys,
                        extract_numbers, MAX_BLOCK, MAX_CANDS_SCORE,
                        MATCH_THRESHOLD, NAMEONLY_THRESHOLD,
                        SHORT_NAME_THRESHOLD, ADDR_REJECT_BELOW)

def f_beta(p, r, beta=0.5):
    if p + r == 0: return 0
    return (1 + beta**2) * p * r / (beta**2 * p + r)

def main():
    T0 = time.time()
    SAMPLE = 15000

    print(f"V3 EVAL — {SAMPLE:,} samples")
    print(f"Config: MATCH={MATCH_THRESHOLD} NAMEONLY={NAMEONLY_THRESHOLD} "
          f"SHORT={SHORT_NAME_THRESHOLD} ADDR_REJ={ADDR_REJECT_BELOW}")

    # Load
    s1 = pd.read_csv(f'{DATA_TRAIN}/train_source1.tsv', sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(f'{DATA_TRAIN}/train_source2.tsv', sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(f'{DATA_TRAIN}/train_source3.tsv', sep='\t', dtype=str).fillna('')
    gt = pd.read_csv(f'{DATA_TRAIN}/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')

    s1_sample = s1.sample(min(SAMPLE, len(s1)), random_state=42).copy()
    gt_sample = gt[gt['source1_entity_id'].isin(set(s1_sample['entity_id']))].copy()

    true_matches = {}
    for _, row in gt_sample.iterrows():
        eid = row['source1_entity_id']
        val = row['matched_entity_ids']
        if pd.notna(val) and str(val).strip():
            true_matches[eid] = set(str(val).split(','))
        else:
            true_matches[eid] = set()

    # Normalize
    for df in [s1_sample, s2, s3]:
        df['nn'] = df['business_name'].apply(norm_name)
        if 'business_address' in df.columns:
            df['an'] = df['business_address'].apply(norm_addr)
        else:
            df['an'] = ''

    # Build index
    print("Building index...")
    t = time.time()
    ref_eid = []; ref_nn = []; ref_an = []
    block_idx = defaultdict(list)
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
    block_idx = {k: v for k, v in block_idx.items() if len(v) <= MAX_BLOCK}
    del s2, s3; gc.collect()
    print(f"  Index: {time.time()-t:.0f}s")

    # Match with V3 logic
    print("Matching...")
    s1_eids = s1_sample['entity_id'].values
    s1_names = s1_sample['nn'].values
    s1_addrs = s1_sample['an'].values
    s1_countries = s1_sample['country'].values

    pred_matches = {}
    stats = {'num_rej': 0, 'addr_rej': 0, 'nameonly_rej': 0, 'short_rej': 0}

    for i in range(len(s1_sample)):
        name = s1_names[i]
        addr = s1_addrs[i]
        country = s1_countries[i]
        s1_eid = s1_eids[i]

        cand_set = set()
        for key in make_keys(name, addr, country):
            if key in block_idx:
                cand_set.update(block_idx[key])

        matched_eids = []
        for ci in list(cand_set)[:MAX_CANDS_SCORE]:
            c_name = ref_nn[ci]
            c_addr = ref_an[ci]

            name_sc = 100.0 if name == c_name else fuzz.token_sort_ratio(name, c_name)
            has_both = bool(addr) and bool(c_addr)
            if has_both:
                addr_sc = fuzz.token_sort_ratio(addr, c_addr)
                combined = 0.6 * name_sc + 0.4 * addr_sc
            else:
                addr_sc = -1
                combined = name_sc

            is_matched = combined >= MATCH_THRESHOLD

            # Post-filter 1: Number mismatch
            if is_matched and has_both:
                n1 = extract_numbers(addr)
                n2 = extract_numbers(c_addr)
                if n1 and n2 and not (n1 & n2):
                    is_matched = False
                    stats['num_rej'] += 1

            # Post-filter 2: Very low address
            if is_matched and has_both and addr_sc < ADDR_REJECT_BELOW:
                is_matched = False
                stats['addr_rej'] += 1

            # Post-filter 3: Name-only stricter
            if is_matched and not has_both:
                if combined < NAMEONLY_THRESHOLD:
                    is_matched = False
                    stats['nameonly_rej'] += 1

            # Post-filter 4: Short name
            if is_matched:
                min_len = min(len(name), len(c_name))
                if min_len < 5 and combined < SHORT_NAME_THRESHOLD:
                    is_matched = False
                    stats['short_rej'] += 1

            if is_matched:
                matched_eids.append(ref_eid[ci])

        pred_matches[s1_eid] = set(matched_eids)

    # Evaluate
    tp = fp = fn = 0
    for s1_eid in true_matches:
        true_set = true_matches[s1_eid]
        pred_set = pred_matches.get(s1_eid, set())
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = f_beta(p, r, 0.5)

    n_pred = sum(1 for v in pred_matches.values() if v)

    print(f"\n{'='*50}")
    print(f"  P={p:.4f}  R={r:.4f}  F0.5={f05:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}")
    print(f"  Entities matched: {n_pred:,} / {len(s1_sample):,}")
    print(f"  Filters: num_rej={stats['num_rej']:,}  addr_rej={stats['addr_rej']:,}  "
          f"nameonly_rej={stats['nameonly_rej']:,}  short_rej={stats['short_rej']:,}")
    print(f"  Time: {(time.time()-T0)/60:.1f}m")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()

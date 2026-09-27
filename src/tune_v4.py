#!/usr/bin/env python3
"""tune_v4.py — Fast threshold sweep for V4 pipeline."""
import os, sys, time, re, gc
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, '/Users/divyanshusingh/Desktop/ML_challenge/code/business_entity_resolution/src')
import test_v4_logic as v4

SAMPLE = 5000
s1 = pd.read_csv(f'{v4.DATA_TRAIN}/train_source1.tsv', sep='\t', dtype=str).fillna('')
s2 = pd.read_csv(f'{v4.DATA_TRAIN}/train_source2.tsv', sep='\t', dtype=str, nrows=500000).fillna('')
s3 = pd.read_csv(f'{v4.DATA_TRAIN}/train_source3.tsv', sep='\t', dtype=str, nrows=500000).fillna('')
gt = pd.read_csv(f'{v4.DATA_TRAIN}/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')

s1_sample = s1.sample(SAMPLE, random_state=42).copy()
gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))
valid_targets = set(s2['entity_id']) | set(s3['entity_id'])
true_pairs = set()
for eid in s1_sample['entity_id']:
    m_str = gt_map.get(eid, '')
    if m_str and pd.notna(m_str):
        for t in str(m_str).split(','):
            t = t.strip()
            if t in valid_targets:
                true_pairs.add((eid, t))

for df in [s1_sample, s2, s3]:
    df['nn'] = df['business_name'].apply(v4.norm_name)
    df['an'] = df['business_address'].apply(v4.norm_addr)
    df['reg'] = df.apply(lambda r: v4.extract_region(r['business_address'], r['country']), axis=1)

ref_eid = []; ref_nn = []; ref_an = []; ref_raw_name = []; ref_raw_addr = []; ref_reg = []
block_idx = defaultdict(list)
for label, df in [('S2', s2), ('S3', s3)]:
    eids = df['entity_id'].values
    names = df['nn'].values
    addrs = df['an'].values
    raw_names = df['business_name'].values
    raw_addrs = df['business_address'].values
    countries = df['country'].values
    regs = df['reg'].values
    for i in range(len(df)):
        pos = len(ref_eid)
        ref_eid.append(eids[i])
        ref_nn.append(names[i])
        ref_an.append(addrs[i])
        ref_raw_name.append(raw_names[i])
        ref_raw_addr.append(raw_addrs[i])
        ref_reg.append(regs[i])
        for key in v4.make_keys(names[i], raw_addrs[i], countries[i]):
            block_idx[key].append(pos)

block_idx = {k: v for k, v in block_idx.items() if len(v) <= 150}
del s2, s3; gc.collect()

_INDIC = re.compile(r'[\u0900-\u0DFF]')
scored_pairs = []
s1_eids = s1_sample['entity_id'].values
s1_names = s1_sample['nn'].values
s1_addrs = s1_sample['an'].values
s1_raw_names = s1_sample['business_name'].values
s1_raw_addrs = s1_sample['business_address'].values
s1_countries = s1_sample['country'].values
s1_regs = s1_sample['reg'].values

for i in range(len(s1_sample)):
    eid = s1_eids[i]
    name = s1_names[i]
    addr = s1_addrs[i]
    raw_name = s1_raw_names[i]
    raw_addr = s1_raw_addrs[i]
    country = s1_countries[i]
    reg = s1_regs[i]
    cand_set = set()
    for key in v4.make_keys(name, raw_addr, country):
        if key in block_idx: cand_set.update(block_idx[key])
    for ci in list(cand_set)[:75]:
        c_reg = ref_reg[ci]
        if reg and c_reg and reg != c_reg: continue
        c_addr = ref_an[ci]
        has_both_addr = bool(addr) and bool(c_addr)
        nums1 = v4.extract_numbers(raw_addr)
        nums2 = v4.extract_numbers(ref_raw_addr[ci])
        if has_both_addr and nums1 and nums2 and not (nums1 & nums2): continue
        c_name = ref_nn[ci]
        if name == c_name: name_sc = 100.0
        else: name_sc = fuzz.token_sort_ratio(name, c_name)
        if has_both_addr:
            addr_sc = fuzz.token_sort_ratio(addr, c_addr)
            comb = 0.6 * name_sc + 0.4 * addr_sc
        else:
            addr_sc = -1
            comb = name_sc
        c_raw_name = ref_raw_name[ci]
        is_indic = has_both_addr and (_INDIC.search(raw_name) or _INDIC.search(c_raw_name)) and addr_sc >= 80 and bool(nums1 & nums2)
        min_len = min(len(name), len(c_name))
        scored_pairs.append((eid, ref_eid[ci], comb, has_both_addr, addr_sc, min_len, is_indic))

print("Scored pairs:", len(scored_pairs))
for th in [70, 72, 74, 76, 78, 80, 82, 84]:
    pred = set()
    for eid, ceid, comb, both_a, a_sc, min_len, is_indic in scored_pairs:
        m = (comb >= th)
        if m and both_a and a_sc < 25: m = False
        if m and not both_a and comb < 88: m = False
        if m and min_len < 5 and comb < 92: m = False
        if not m and is_indic: m = True
        if m: pred.add((eid, ceid))
    tp = len(pred & true_pairs)
    fp = len(pred - true_pairs)
    fn = len(true_pairs - pred)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = v4.f_beta(p, r, 0.5)
    print(f'Th={th:2d} | P={p:.4f} ({p*100:5.1f}%) | R={r:.4f} ({r*100:5.1f}%) | F0.5={f05:.4f} | TP={tp} FP={fp}')

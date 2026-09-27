#!/usr/bin/env python3
"""
test_v4_logic.py — Benchmark V4 logic on 10K training samples against ground truth.
Calculates exact Precision, Recall, and F0.5.
"""
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

# ── Regions ────────────────────────────────────────────────────────────────
US_STATES = frozenset([
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY'
])

INDIA_STATES = {
    'maharashtra': 'MH', 'karnataka': 'KA', 'tamil nadu': 'TN', 'tamilnadu': 'TN',
    'delhi': 'DL', 'gujarat': 'GJ', 'uttar pradesh': 'UP', 'west bengal': 'WB',
    'telangana': 'TG', 'andhra pradesh': 'AP', 'rajasthan': 'RJ', 'kerala': 'KL',
    'madhya pradesh': 'MP', 'haryana': 'HR', 'bihar': 'BR', 'punjab': 'PB',
    'odisha': 'OR', 'orissa': 'OR', 'assam': 'AS', 'jharkhand': 'JH',
    'uttarakhand': 'UK', 'goa': 'GA', 'himachal pradesh': 'HP', 'chhattisgarh': 'CG'
}
INDIA_CODES = frozenset(INDIA_STATES.values())

FRANCE_REGIONS = [
    'nouvelle-aquitaine', 'hauts-de-france', 'pays de la loire', 'ile-de-france',
    'auvergne-rhone-alpes', 'occitanie', 'grand est', 'provence-alpes-cote d azur',
    'bretagne', 'normandie', 'bourgogne-franche-comte', 'centre-val de loire', 'corse'
]

def extract_region(raw_addr, country):
    if not raw_addr: return None
    if country == 'US':
        tokens = re.findall(r'\b[A-Za-z]{2}\b', str(raw_addr).upper())
        for t in reversed(tokens):
            if t in US_STATES: return t
    elif country == 'India':
        a = str(raw_addr).lower()
        for name, code in INDIA_STATES.items():
            if name in a: return code
        tokens = re.findall(r'\b[A-Za-z]{2}\b', str(raw_addr).upper())
        for t in reversed(tokens):
            if t in INDIA_CODES: return t
    elif country == 'France':
        a = str(raw_addr).lower().replace('î', 'i').replace('ô', 'o').replace("'", " ")
        for r in FRANCE_REGIONS:
            if r in a: return r
    return None

# ── Normalization ─────────────────────────────────────────────────────────
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
_NUMBERS = re.compile(r'\b\d+\b')
_INDIC = re.compile(r'[\u0900-\u0DFF]')
_STOP_ADDR = frozenset(['the', 'of', 'and', 'for', 'in', 'at', 'to', 'a', 'an', 'is', 'it', 'or', 'by', 'on', 'no', 'so', 'do',
                        'st', 'street', 'ave', 'avenue', 'rd', 'road', 'dr', 'drive', 'ln', 'lane', 'blvd', 'boulevard',
                        'hwy', 'highway', 'pkwy', 'parkway', 'house', 'plot', 'flat', 'unit', 'suite', 'apt', 'room', 'floor',
                        'near', 'opp', 'opposite', 'behind', 'beside', 'phase', 'sec', 'sector', 'block'])

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

def extract_numbers(text):
    return set(_NUMBERS.findall(str(text or '')))

def make_keys(name_n, addr_raw, country):
    ws = meaningful_words(name_n)
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
    if ws and len(ws[0]) >= 4:
        vowels = frozenset('aeiou')
        cons = ''.join(c for c in ws[0] if c not in vowels and c.isalpha())[:6]
        if len(cons) >= 3:
            keys.append(f"C|{country}|{cons}")
    # Address key
    if addr_raw:
        words = _PUNCT.sub(' ', str(addr_raw).lower()).split()
        nums = [w for w in words if w.isdigit()]
        m_addr = [w for w in words if not w.isdigit() and len(w) > 2 and w not in _STOP_ADDR]
        if nums and m_addr:
            keys.append(f"AN|{country}|{nums[0]}|{m_addr[0]}")
    return keys

def f_beta(p, r, beta=0.5):
    if p + r == 0: return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)

def main():
    SAMPLE = 5000
    print(f"Loading data (eval on {SAMPLE} samples)...")
    s1 = pd.read_csv(f'{DATA_TRAIN}/train_source1.tsv', sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(f'{DATA_TRAIN}/train_source2.tsv', sep='\t', dtype=str, nrows=500000).fillna('')
    s3 = pd.read_csv(f'{DATA_TRAIN}/train_source3.tsv', sep='\t', dtype=str, nrows=500000).fillna('')
    gt = pd.read_csv(f'{DATA_TRAIN}/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')

    s1_sample = s1.sample(SAMPLE, random_state=42).copy()
    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))

    # Filter true matches to those present in s2/s3 subset
    valid_targets = set(s2['entity_id']) | set(s3['entity_id'])
    true_pairs = set()
    for eid in s1_sample['entity_id']:
        m_str = gt_map.get(eid, '')
        if m_str and pd.notna(m_str):
            for t in str(m_str).split(','):
                t = t.strip()
                if t in valid_targets:
                    true_pairs.add((eid, t))

    print(f"True positive pairs in sample: {len(true_pairs):,}")

    for df in [s1_sample, s2, s3]:
        df['nn'] = df['business_name'].apply(norm_name)
        df['an'] = df['business_address'].apply(norm_addr)
        df['reg'] = df.apply(lambda r: extract_region(r['business_address'], r['country']), axis=1)

    print("Building blocking index...")
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
            for key in make_keys(names[i], raw_addrs[i], countries[i]):
                block_idx[key].append(pos)

    block_idx = {k: v for k, v in block_idx.items() if len(v) <= 150}
    del s2, s3; gc.collect()

    print(f"Index built ({len(ref_eid):,} entities, {len(block_idx):,} blocks). Evaluating...")

    MATCH_THRESHOLD = 84
    pred_pairs = set()

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
        for key in make_keys(name, raw_addr, country):
            if key in block_idx:
                cand_set.update(block_idx[key])

        for ci in list(cand_set)[:50]:
            c_name = ref_nn[ci]
            c_addr = ref_an[ci]
            c_raw_name = ref_raw_name[ci]
            c_raw_addr = ref_raw_addr[ci]
            c_reg = ref_reg[ci]
            c_eid = ref_eid[ci]

            # 1. State / Region check
            if reg and c_reg and reg != c_reg:
                continue

            # 2. Number check
            has_both_addr = bool(addr) and bool(c_addr)
            nums1 = extract_numbers(raw_addr)
            nums2 = extract_numbers(c_raw_addr)
            if has_both_addr and nums1 and nums2 and not (nums1 & nums2):
                continue

            # 3. Name scoring
            if name == c_name:
                name_sc = 100.0
            else:
                name_sc = fuzz.token_sort_ratio(name, c_name)

            if has_both_addr:
                addr_sc = fuzz.token_sort_ratio(addr, c_addr)
                combined = 0.6 * name_sc + 0.4 * addr_sc
            else:
                addr_sc = -1
                combined = name_sc

            matched = combined >= MATCH_THRESHOLD

            # Precision filters
            if matched and has_both_addr and addr_sc < 30:
                matched = False
            if matched and not has_both_addr and combined < 90:
                matched = False
            if matched and min(len(name), len(c_name)) < 5 and combined < 95:
                matched = False

            # Indic script recovery
            if not matched and has_both_addr and (_INDIC.search(raw_name) or _INDIC.search(c_raw_name)):
                if addr_sc >= 82 and (nums1 & nums2):
                    matched = True

            if matched:
                pred_pairs.add((eid, c_eid))

    # Calculate metrics
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = f_beta(p, r, 0.5)

    print("=" * 60)
    print(f"  EVALUATION RESULTS:")
    print(f"  True Positives:  {tp:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  Precision:       {p:.4f} ({p*100:.1f}%)")
    print(f"  Recall:          {r:.4f} ({r*100:.1f}%)")
    print(f"  ★ F0.5 Score:    {f05:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main()

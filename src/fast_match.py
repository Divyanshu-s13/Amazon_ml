#!/usr/bin/env python3
"""
fast_match.py — V4: High-Precision Hash-Based Entity Resolution Pipeline.

Key improvements:
  1. Geographic state/region filtering (US states, Indian states, French regions)
     eliminates cross-region false positives.
  2. Address street number mismatch rejection ("123 Main" ≠ "456 Main").
  3. Low address similarity rejection (addr_sc < 25 rejected).
  4. Indic script recovery for cross-script matches in India (English vs Hindi/Tamil).
  5. Short name penalty and missing-address strict thresholds.
  6. 100% candidate list consistency (all matched IDs guaranteed in candidate_pairs).
"""

import os, sys, time, re, gc, shutil, subprocess
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

# ── Configuration ─────────────────────────────────────────────────────────
MATCH_THRESHOLD = 82       # Tuned threshold: P=89.1%, F0.5=0.685
NAMEONLY_THRESHOLD = 88    # Stricter threshold when address is missing
SHORT_NAME_THRESHOLD = 92  # Even stricter for very short names (<5 chars)
ADDR_REJECT_BELOW = 25     # Reject match if addr similarity is below this
CANDIDATE_THRESHOLD = 40   # Minimum score for candidate_pairs.tsv
MAX_BLOCK = 150            # Prune blocking keys larger than this
MAX_CANDS_SCORE = 75       # Max candidates to score per S1 entity
MAX_CANDS_OUTPUT = 25      # Candidates written to candidate_pairs.tsv

# ── Path resolution ───────────────────────────────────────────────────────
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

DATA_TEST = os.path.join(ROOT, 'student_resource/dataset/test')
OUTPUT = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT, exist_ok=True)

# ── Region / State Extraction ─────────────────────────────────────────────
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

# ── Text normalization ────────────────────────────────────────────────────
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
    """Extract number tokens from text for address matching."""
    return set(_NUMBERS.findall(str(text or '')))


# ── Blocking key generation ──────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def main():
    T0 = time.time()

    print("=" * 60)
    print("  FAST ENTITY RESOLUTION PIPELINE V4")
    print("  Strategy: High-Precision Geo-Gated Hash Blocking")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print(f"\n[1/6] Loading test data...")
    t = time.time()
    s1 = pd.read_csv(f'{DATA_TEST}/test_source1.tsv', sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(f'{DATA_TEST}/test_source2.tsv', sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(f'{DATA_TEST}/test_source3.tsv', sep='\t', dtype=str).fillna('')
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  ({time.time()-t:.0f}s)")

    # ── Step 2: Normalize and extract regions ──────────────────────────────
    print(f"\n[2/6] Normalizing and extracting geographic regions...")
    t = time.time()
    for df in [s1, s2, s3]:
        df['nn'] = df['business_name'].apply(norm_name)
        if 'business_address' in df.columns:
            df['an'] = df['business_address'].apply(norm_addr)
            df['reg'] = df.apply(lambda r: extract_region(r['business_address'], r['country']), axis=1)
        else:
            df['an'] = ''
            df['reg'] = None
    print(f"  Done in {time.time()-t:.0f}s")

    # ── Step 3: Build blocking index ──────────────────────────────────────
    n_ref = len(s2) + len(s3)
    print(f"\n[3/6] Building index for S2+S3 ({n_ref:,} entities)...")
    t = time.time()

    ref_eid = []; ref_nn = []; ref_an = []
    ref_raw_name = []; ref_raw_addr = []; ref_reg = []
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
            if (i + 1) % 2_000_000 == 0:
                print(f"    {label}: {i+1:,} / {len(df):,} ({time.time()-t:.0f}s)")
        print(f"  {label}: {len(df):,} indexed ({time.time()-t:.0f}s)")

    n_before = len(block_idx)
    block_idx = {k: v for k, v in block_idx.items() if len(v) <= MAX_BLOCK}
    print(f"  Pruned {n_before - len(block_idx):,} large blocks")
    print(f"  Active: {len(block_idx):,} blocks, {len(ref_eid):,} entities")
    print(f"  Built in {time.time()-t:.0f}s")

    del s2, s3; gc.collect()

    # ── Step 4: Match S1 entities ─────────────────────────────────────────
    print(f"\n[4/6] Matching {len(s1):,} S1 entities...")
    t = time.time()

    s1_eids = s1['entity_id'].values
    s1_names = s1['nn'].values
    s1_addrs = s1['an'].values
    s1_raw_names = s1['business_name'].values
    s1_raw_addrs = s1['business_address'].values
    s1_countries = s1['country'].values
    s1_regs = s1['reg'].values

    all_candidates = {}
    all_matches = {}
    n_scored = 0

    # Filter counters
    n_reg_rejected = 0
    n_num_rejected = 0
    n_addr_rejected = 0
    n_nameonly_rejected = 0
    n_short_rejected = 0
    n_indic_recovered = 0

    for i in range(len(s1)):
        s1_eid = s1_eids[i]
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

        if not cand_set:
            all_candidates[s1_eid] = []
            all_matches[s1_eid] = []
        else:
            scored = []
            nums1 = extract_numbers(raw_addr)
            has_s1_addr = bool(addr)

            for ci in list(cand_set)[:MAX_CANDS_SCORE]:
                c_reg = ref_reg[ci]
                # Filter 1: Cross-region mismatch
                if reg and c_reg and reg != c_reg:
                    n_reg_rejected += 1
                    continue

                c_addr = ref_an[ci]
                c_raw_addr = ref_raw_addr[ci]
                has_both_addr = has_s1_addr and bool(c_addr)

                # Filter 2: Street number mismatch
                nums2 = extract_numbers(c_raw_addr)
                if has_both_addr and nums1 and nums2 and not (nums1 & nums2):
                    n_num_rejected += 1
                    continue

                c_name = ref_nn[ci]
                c_raw_name = ref_raw_name[ci]

                # Scoring
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

                # Filter 3: Low address rejection
                if matched and has_both_addr and addr_sc < ADDR_REJECT_BELOW:
                    matched = False
                    n_addr_rejected += 1

                # Filter 4: Stricter threshold when address is missing
                if matched and not has_both_addr and combined < NAMEONLY_THRESHOLD:
                    matched = False
                    n_nameonly_rejected += 1

                # Filter 5: Short name penalty
                min_len = min(len(name), len(c_name))
                if matched and min_len < 5 and combined < SHORT_NAME_THRESHOLD:
                    matched = False
                    n_short_rejected += 1

                # Filter 6: Indic script cross-language recovery
                if not matched and has_both_addr and (_INDIC.search(raw_name) or _INDIC.search(c_raw_name)):
                    if addr_sc >= 80 and (nums1 & nums2):
                        matched = True
                        combined = max(combined, 85.0)
                        n_indic_recovered += 1

                scored.append((ref_eid[ci], combined, matched))
                n_scored += 1

            scored.sort(key=lambda x: -x[1])

            # Ensure candidate consistency:
            match_list = [e for e, sc, m in scored if m]
            cand_list = [e for e, sc, _ in scored if sc >= CANDIDATE_THRESHOLD][:MAX_CANDS_OUTPUT]
            cand_set_out = set(cand_list)
            for m in match_list:
                if m not in cand_set_out:
                    cand_list.append(m)
                    cand_set_out.add(m)

            all_candidates[s1_eid] = cand_list
            all_matches[s1_eid] = match_list

        if (i + 1) % 200_000 == 0:
            el = time.time() - t
            rate = (i + 1) / el
            eta = (len(s1) - i - 1) / rate
            n_m = sum(1 for v in all_matches.values() if v)
            print(f"  {i+1:>10,} / {len(s1):,} | {el:>6.0f}s | "
                  f"ETA: {eta/60:>5.1f}m | matched: {n_m:,}")

    el = time.time() - t
    n_matched = sum(1 for v in all_matches.values() if v)
    n_cand = sum(1 for v in all_candidates.values() if v)
    total_match_pairs = sum(len(v) for v in all_matches.values())

    print(f"\n  Matching complete in {el:.0f}s ({el/60:.1f} min)")
    print(f"  S1 with candidates: {n_cand:,} / {len(s1):,}")
    print(f"  S1 with matches:    {n_matched:,} / {len(s1):,}")
    print(f"  Total match pairs:  {total_match_pairs:,}")
    print(f"  Total scored:       {n_scored:,}")
    print(f"\n  Filters applied:")
    print(f"    Region mismatch rejected:  {n_reg_rejected:,}")
    print(f"    Number mismatch rejected:  {n_num_rejected:,}")
    print(f"    Low address rejected:      {n_addr_rejected:,}")
    print(f"    Name-only rejected:        {n_nameonly_rejected:,}")
    print(f"    Short name rejected:       {n_short_rejected:,}")
    print(f"    Indic script recovered:    {n_indic_recovered:,}")

    # ── Step 5: Write output ──────────────────────────────────────────────
    print(f"\n[5/6] Writing output files...")

    cp_path = os.path.join(OUTPUT, 'candidate_pairs.tsv')
    mr_path = os.path.join(OUTPUT, 'matching_results.tsv')

    with open(cp_path, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for eid in s1_eids:
            cands = all_candidates.get(eid, [])
            f.write(f"{eid}\t{','.join(cands)}\n")
    print(f"  Written: {cp_path}")

    with open(mr_path, 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for eid in s1_eids:
            matches = all_matches.get(eid, [])
            f.write(f"{eid}\t{','.join(matches)}\n")
    print(f"  Written: {mr_path}")

    shutil.copy(cp_path, os.path.join(OUTPUT, 'candidate_pairs_test.tsv'))

    # ── Step 6: Validate and package ──────────────────────────────────────
    print(f"\n[6/6] Validating and packaging...")

    validate_script = os.path.join(ROOT, 'student_resource/utils/validate_submission.py')
    if os.path.exists(validate_script):
        result = subprocess.run(
            [sys.executable, validate_script,
             '--matching', mr_path,
             '--candidate', cp_path,
             '--test-dir', DATA_TEST],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ⚠️ {result.stderr}")

    zip_path = os.path.join(ROOT, 'Divyanshu_submission.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    subprocess.run(['zip', '-j', zip_path, mr_path, cp_path], capture_output=True)
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  ZIP: {zip_path} ({zip_size:.1f} MB)")

    total = (time.time() - T0) / 60
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE V4 COMPLETE in {total:.1f} minutes")
    print(f"  📦 Submission: {zip_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

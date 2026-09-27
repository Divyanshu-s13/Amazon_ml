#!/usr/bin/env python3
"""
fast_match.py — Ultra-fast entity resolution using hash-based blocking + RapidFuzz.

Strategy:
  1. Multi-key hash blocking (exact name, word subsets, address prefix)
     → O(1) lookups instead of O(N×M) TF-IDF cosine similarity
  2. RapidFuzz token_sort_ratio scoring on candidates
  3. Conservative thresholding for high precision (F0.5 metric)

Target: all 1.73M test entities in ~20-30 minutes on a Mac.
"""

import os, sys, time, re, gc, shutil, subprocess
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

# ── Configuration ─────────────────────────────────────────────────────────
MATCH_THRESHOLD = 82       # Combined score ≥ this → final match
CANDIDATE_THRESHOLD = 50   # Combined score ≥ this → candidate pair
MAX_BLOCK = 150            # Prune blocks larger than this
MAX_CANDS_SCORE = 30       # Max candidates to score per S1 entity

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


def norm_name(text):
    """Normalize business name: lowercase, remove legal suffixes and punctuation."""
    t = str(text or '').lower()
    t = _LEGAL.sub('', t)
    t = _PUNCT.sub(' ', t)
    return _MULTI.sub(' ', t).strip()


def norm_addr(text):
    """Normalize address: lowercase, standardize abbreviations, remove punctuation."""
    t = str(text or '').lower()
    t = _PUNCT.sub(' ', t)
    words = t.split()
    words = [_ADDR_MAP.get(w, w) for w in words]
    return ' '.join(words).strip()


def meaningful_words(text):
    """Extract meaningful words (>1 char, not stopwords)."""
    return [w for w in text.split() if len(w) > 1 and w not in _STOP]


# ── Blocking key generation ──────────────────────────────────────────────
def make_keys(name_n, addr_n, country):
    """Generate multiple blocking keys for hash-based candidate retrieval."""
    ws = meaningful_words(name_n)
    aw = meaningful_words(addr_n)
    keys = []

    # K1: Exact normalized name (highest precision)
    if name_n and len(name_n) > 2:
        keys.append(f"E|{country}|{name_n}")

    # K2: First 2 meaningful words of name
    if len(ws) >= 2:
        keys.append(f"W2|{country}|{ws[0]}|{ws[1]}")

    # K3: Sorted first 3 meaningful words (order-independent)
    if len(ws) >= 2:
        sw = sorted(set(ws[:4]))
        keys.append(f"S|{country}|{'|'.join(sw[:3])}")

    # K4: First meaningful word (only if specific enough)
    if ws and len(ws[0]) >= 5:
        keys.append(f"W1|{country}|{ws[0]}")

    # K5: First + last word (catches reorderings/insertions)
    if len(ws) >= 3:
        keys.append(f"FL|{country}|{ws[0]}|{ws[-1]}")

    # K6: Address prefix — first 3 meaningful words of address
    if len(aw) >= 3:
        keys.append(f"A|{country}|{aw[0]}|{aw[1]}|{aw[2]}")

    return keys


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def main():
    T0 = time.time()

    print("=" * 60)
    print("  FAST ENTITY RESOLUTION PIPELINE")
    print("  Strategy: Hash Blocking + RapidFuzz Scoring")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print(f"\n[1/6] Loading test data...")
    t = time.time()
    s1 = pd.read_csv(f'{DATA_TEST}/test_source1.tsv', sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(f'{DATA_TEST}/test_source2.tsv', sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(f'{DATA_TEST}/test_source3.tsv', sep='\t', dtype=str).fillna('')
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  ({time.time()-t:.0f}s)")

    # ── Step 2: Normalize ─────────────────────────────────────────────────
    print(f"\n[2/6] Normalizing business names and addresses...")
    t = time.time()
    for df in [s1, s2, s3]:
        df['nn'] = df['business_name'].apply(norm_name)
        if 'business_address' in df.columns:
            df['an'] = df['business_address'].apply(norm_addr)
        else:
            df['an'] = ''
    print(f"  Done in {time.time()-t:.0f}s")

    # ── Step 3: Build blocking index for S2+S3 ───────────────────────────
    n_ref = len(s2) + len(s3)
    print(f"\n[3/6] Building blocking index for S2+S3 ({n_ref:,} entities)...")
    t = time.time()

    # Reference entity arrays (memory-efficient parallel arrays)
    ref_eid = []
    ref_nn = []
    ref_an = []

    # Inverted index: string_key → list of integer indices
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

            if (i + 1) % 2_000_000 == 0:
                print(f"    {label}: {i+1:,} / {len(df):,} ({time.time()-t:.0f}s)")

        print(f"  {label}: {len(df):,} entities indexed ({time.time()-t:.0f}s)")

    # Prune oversized blocks (not discriminative)
    n_before = len(block_idx)
    block_idx = {k: v for k, v in block_idx.items() if len(v) <= MAX_BLOCK}
    n_pruned = n_before - len(block_idx)
    print(f"  Pruned {n_pruned:,} oversized blocks (>{MAX_BLOCK})")
    print(f"  Active blocks: {len(block_idx):,}")
    print(f"  Reference entities: {len(ref_eid):,}")
    print(f"  Index built in {time.time()-t:.0f}s")

    # Free source DataFrames
    del s2, s3
    gc.collect()

    # ── Step 4: Match S1 entities ─────────────────────────────────────────
    print(f"\n[4/6] Matching {len(s1):,} S1 entities...")
    t = time.time()

    s1_eids = s1['entity_id'].values
    s1_names = s1['nn'].values
    s1_addrs = s1['an'].values
    s1_countries = s1['country'].values

    all_candidates = {}  # s1_eid → list of ref_eids
    all_matches = {}     # s1_eid → list of ref_eids
    n_scored = 0

    for i in range(len(s1)):
        s1_eid = s1_eids[i]
        name = s1_names[i]
        addr = s1_addrs[i]
        country = s1_countries[i]

        # Collect candidate indices from all blocking keys
        cand_set = set()
        for key in make_keys(name, addr, country):
            if key in block_idx:
                cand_set.update(block_idx[key])

        if not cand_set:
            all_candidates[s1_eid] = []
            all_matches[s1_eid] = []
        else:
            # Score candidates with RapidFuzz
            scored = []
            for ci in list(cand_set)[:MAX_CANDS_SCORE]:
                c_name = ref_nn[ci]
                c_addr = ref_an[ci]

                # Name similarity
                if name == c_name:
                    name_sc = 100.0
                else:
                    name_sc = fuzz.token_sort_ratio(name, c_name)

                # Address similarity (when both available)
                if addr and c_addr:
                    addr_sc = fuzz.token_sort_ratio(addr, c_addr)
                    combined = 0.6 * name_sc + 0.4 * addr_sc
                else:
                    combined = name_sc

                scored.append((ref_eid[ci], combined, name_sc))
                n_scored += 1

            scored.sort(key=lambda x: -x[1])

            all_candidates[s1_eid] = [e for e, sc, _ in scored if sc >= CANDIDATE_THRESHOLD][:10]
            all_matches[s1_eid] = [e for e, sc, nsc in scored if sc >= MATCH_THRESHOLD]

        if (i + 1) % 200_000 == 0:
            el = time.time() - t
            rate = (i + 1) / el
            eta = (len(s1) - i - 1) / rate
            n_m = sum(1 for v in all_matches.values() if v)
            print(f"  {i+1:>10,} / {len(s1):,} | {el:>6.0f}s | "
                  f"ETA: {eta/60:>5.1f}m | matched: {n_m:,} | scored: {n_scored:,}")

    el = time.time() - t
    n_matched = sum(1 for v in all_matches.values() if v)
    n_cand = sum(1 for v in all_candidates.values() if v)
    total_match_pairs = sum(len(v) for v in all_matches.values())

    print(f"\n  Matching complete in {el:.0f}s ({el/60:.1f} min)")
    print(f"  S1 with candidates: {n_cand:,} / {len(s1):,}")
    print(f"  S1 with matches:    {n_matched:,} / {len(s1):,}")
    print(f"  Total match pairs:  {total_match_pairs:,}")
    print(f"  Total scored:       {n_scored:,}")

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
            print(f"  ⚠️  Validation error: {result.stderr}")

    # Create submission zip
    zip_path = os.path.join(ROOT, 'Divyanshu_submission.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    subprocess.run(
        ['zip', '-j', zip_path, mr_path, cp_path],
        capture_output=True
    )
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  ZIP: {zip_path} ({zip_size:.1f} MB)")

    total = (time.time() - T0) / 60
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE COMPLETE in {total:.1f} minutes")
    print(f"  📦 Submission: {zip_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

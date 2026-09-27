"""
blocking.py — Scalable TF-IDF candidate generation for business entity resolution.

Key improvements over previous version:
  • Better text normalization (legal suffixes, address abbreviations)
  • Memory-efficient: S2 and S3 processed separately (halves peak RAM)
  • float32 sparse matrices (further halves RAM)
  • n_threads=-1 for multi-core sparse matmul
  • No default limit for test mode → processes ALL 1.7M test entities
  • Random training sample for diversity
"""
import re
import os
import gc
import shutil
import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import sparse_dot_topn
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Text normalization ─────────────────────────────────────────────────────
LEGAL_PATTERNS = [
    (r'\bltd\.?\b', 'limited'),
    (r'\bllc\.?\b', 'llc'),
    (r'\binc\.?\b', 'incorporated'),
    (r'\bcorp\.?\b', 'corporation'),
    (r'\bco\.?\b', 'company'),
    (r'\bpvt\.?\b', 'private'),
    (r'\bgmbh\b', 'gmbh'),
    (r'\bsas\b', 'sas'),
    (r'\bplc\b', 'plc'),
    (r'\bbv\b', 'bv'),
    (r'\bnv\b', 'nv'),
    (r'\bsrl\b', 'srl'),
    (r'\bspa\b', 'spa'),
]

ADDR_PATTERNS = [
    (r'\bst\.?\b', 'street'),
    (r'\bave\.?\b', 'avenue'),
    (r'\bblvd\.?\b', 'boulevard'),
    (r'\brd\.?\b', 'road'),
    (r'\bdr\.?\b', 'drive'),
    (r'\bln\.?\b', 'lane'),
    (r'\bct\.?\b', 'court'),
    (r'\bpl\.?\b', 'place'),
    (r'\bpkwy\.?\b', 'parkway'),
    (r'\bapt\.?\b', 'apartment'),
    (r'\bste\.?\b', 'suite'),
    (r'\bflr\.?\b', 'floor'),
    (r'\bhwy\.?\b', 'highway'),
]

_PUNCT = re.compile(r'[^\w\s]')
_SPACE = re.compile(r'\s+')


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


def make_text(name, address):
    return normalize_name(name) + ' ' + normalize_address(address)


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


# ── Core blocking ──────────────────────────────────────────────────────────
def create_blocking(mode='train', limit=None, top_n=5, threshold=0.20):
    dataset_dir, output_dir = find_paths()
    base_path = os.path.join(dataset_dir, mode)
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*55}")
    print(f"  Blocking mode={mode}  top_n={top_n}  threshold={threshold}")
    print(f"{'='*55}")

    # ── Load datasets ──────────────────────────────────────
    print(f"\nLoading datasets from {base_path} ...")
    df1 = pd.read_csv(f'{base_path}/{mode}_source1.tsv', sep='\t', dtype=str).fillna('')
    df2 = pd.read_csv(f'{base_path}/{mode}_source2.tsv', sep='\t', dtype=str).fillna('')
    df3 = pd.read_csv(f'{base_path}/{mode}_source3.tsv', sep='\t', dtype=str).fillna('')
    print(f"  S1:{len(df1):>10,}  S2:{len(df2):>10,}  S3:{len(df3):>10,}")

    # ── Normalize text ────────────────────────────────────
    print("  Normalizing text ...")
    for df in (df1, df2, df3):
        df['text'] = [make_text(n, a)
                      for n, a in zip(df['business_name'], df['business_address'])]

    # ── Determine search set for S1 ────────────────────────
    if mode == 'train':
        # Random sample for diverse training data
        train_limit = limit if limit is not None else 6000
        df1_search = df1.sample(min(train_limit, len(df1)), random_state=42).copy()
        print(f"  Training search: {len(df1_search):,} S1 entities (random sample)")
    else:
        # Test mode: ALWAYS process all S1 entities
        if limit is not None:
            df1_search = df1.head(limit).copy()
            print(f"  Test search (limited): {len(df1_search):,} S1 entities")
        else:
            df1_search = df1
            print(f"  Test search (FULL): {len(df1_search):,} S1 entities")

    countries = [c for c in df1_search['country'].unique() if c]
    results = defaultdict(list)
    BATCH = 100_000  # S1 batch size for matrix multiply (larger = fewer loop iters)

    for country in countries:
        c1 = df1_search[df1_search['country'] == country].reset_index(drop=True)
        c2 = df2[df2['country'] == country].reset_index(drop=True)
        c3 = df3[df3['country'] == country].reset_index(drop=True)

        print(f"\n{'─'*45}")
        print(f"  Country: {country}  |  S1={len(c1):,}  S2={len(c2):,}  S3={len(c3):,}")
        if not len(c1):
            continue

        # ── Fit TF-IDF vectorizer ──────────────────────────
        print("  Fitting TF-IDF vectorizer ...")
        vect = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        corpus_sample = []
        if len(c2):
            corpus_sample += list(c2['text'].sample(min(50_000, len(c2)), random_state=42))
        if len(c3):
            corpus_sample += list(c3['text'].sample(min(50_000, len(c3)), random_state=42))
        vect.fit(corpus_sample)
        del corpus_sample

        s1_ids = c1['entity_id'].to_numpy()
        num_batches = int(np.ceil(len(c1) / BATCH))

        # ── Match S1 → S2 ─────────────────────────────────
        if len(c2):
            print(f"  Transforming S2 ({len(c2):,} rows) ...", end=' ', flush=True)
            t0 = time.time()
            M2_T = vect.transform(c2['text']).T.tocsr()
            print(f"{time.time()-t0:.1f}s")
            s2_ids = c2['entity_id'].to_numpy()
            for i in range(num_batches):
                M1 = vect.transform(c1['text'].iloc[i*BATCH:(i+1)*BATCH])
                hits = sparse_dot_topn.sp_matmul_topn(
                    M1, M2_T, top_n=top_n, threshold=threshold, n_threads=-1)
                rows, cols = hits.nonzero()
                for r, c_ in zip(rows, cols):
                    results[s1_ids[i*BATCH + r]].append(s2_ids[c_])
                if (i+1) % 10 == 0 or (i+1) == num_batches:
                    print(f"    S2 batch {i+1}/{num_batches}", flush=True)
            del M2_T; gc.collect()

        # ── Match S1 → S3 ─────────────────────────────────
        if len(c3):
            print(f"  Transforming S3 ({len(c3):,} rows) ...", end=' ', flush=True)
            t0 = time.time()
            M3_T = vect.transform(c3['text']).T.tocsr()
            print(f"{time.time()-t0:.1f}s")
            s3_ids = c3['entity_id'].to_numpy()
            for i in range(num_batches):
                M1 = vect.transform(c1['text'].iloc[i*BATCH:(i+1)*BATCH])
                hits = sparse_dot_topn.sp_matmul_topn(
                    M1, M3_T, top_n=top_n, threshold=threshold, n_threads=-1)
                rows, cols = hits.nonzero()
                for r, c_ in zip(rows, cols):
                    results[s1_ids[i*BATCH + r]].append(s3_ids[c_])
                if (i+1) % 10 == 0 or (i+1) == num_batches:
                    print(f"    S3 batch {i+1}/{num_batches}", flush=True)
            del M3_T; gc.collect()

        del c2, c3, vect; gc.collect()

    # ── Save output (ALL S1 entities must be present) ──────
    out_path = os.path.join(output_dir, f'candidate_pairs_{mode}.tsv')
    print(f"\nSaving → {out_path}")
    cand_counts = []
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for eid in df1['entity_id']:
            cands = results.get(eid, [])
            seen = set(); deduped = []
            for x in cands:
                if x not in seen:
                    seen.add(x); deduped.append(x)
            f.write(f'{eid}\t{",".join(deduped)}\n')
            cand_counts.append(len(deduped))

    total_ents = len(cand_counts)
    non_empty = sum(1 for c in cand_counts if c > 0)
    avg_cands = sum(cand_counts) / max(1, total_ents)
    print(f"  Saved {total_ents:,} rows | non-empty: {non_empty:,} | avg candidates: {avg_cands:.1f}")

    if mode == 'test':
        canon = os.path.join(output_dir, 'candidate_pairs.tsv')
        shutil.copy(out_path, canon)
        print(f"  Copied to {canon}")

    # ── Evaluate recall on training split ─────────────────
    if mode == 'train':
        gt_path = os.path.join(base_path, 'train_ground_truth.tsv')
        if os.path.exists(gt_path):
            print("\nEvaluating blocking recall ...")
            gt = pd.read_csv(gt_path, sep='\t')
            proc = set(df1_search['entity_id'])
            gt = gt[gt['source1_entity_id'].isin(proc)]
            found = total = 0
            for _, row in gt.iterrows():
                val = row['matched_entity_ids']
                if pd.isna(val) or not str(val).strip():
                    continue
                true_ids = set(str(val).split(','))
                cands = set(results.get(row['source1_entity_id'], []))
                found += len(true_ids & cands)
                total += len(true_ids)
            if total:
                print(f"  Blocking Recall = {found/total:.4f}  ({found}/{total})")

    print(f"\n  Total elapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='train', choices=['train', 'test'])
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit test S1 for quick testing. Leave unset for full run.')
    ap.add_argument('--top_n', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=0.20)
    args = ap.parse_args()
    create_blocking(args.mode, args.limit, args.top_n, args.threshold)

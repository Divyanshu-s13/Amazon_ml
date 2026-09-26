#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution Pipeline
Team: Divyanshu, Shivam, Vansh, Jatin

Strategy:
  1. Normalize business names & addresses (lowercase, remove punctuation,
     expand common abbreviations, strip legal suffixes).
  2. Build a Character-Trigram Inverted Index over all Source-2 and Source-3
     records so we can instantly retrieve plausible candidates for each S1 entity.
  3. Rerank candidates using a combination of:
       - Trigram Jaccard similarity on name
       - Token Jaccard similarity on name
       - Shared token count on address
       - Country exact-match guard
  4. Apply a precision-tuned threshold (F_0.5 weights precision 2x over recall).
  5. Write out candidate_pairs.tsv and matching_results.tsv in the correct format.
"""

import csv
import collections
import os
import re
import string
import sys
import time
import multiprocessing as mp
from itertools import islice

# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────
BASE     = '/Users/divyanshusingh/Desktop/ML_challenge/student_resource/dataset'
OUT_DIR  = '/Users/divyanshusingh/Desktop/ML_challenge/output'
os.makedirs(OUT_DIR, exist_ok=True)

# Blocking: keep top-K candidates per S1 entity after trigram retrieval
TOP_K_CANDIDATES = 20

# Final matching thresholds (tuned for high precision on F_0.5)
NAME_TRIGRAM_THRESHOLD = 0.30   # minimum trigram Jaccard to be a candidate
MATCH_THRESHOLD_HIGH   = 0.80   # accept on name Jaccard alone (very confident)
MATCH_THRESHOLD_LOW    = 0.60   # accept if name + address both exceed this
ADDR_THRESHOLD         = 0.35   # minimum address token Jaccard for MATCH_THRESHOLD_LOW path

# Parallelism
N_WORKERS = max(1, os.cpu_count() - 1)  # leave 1 core for the OS

# ──────────────────────────────────────────
# Abbreviation / normalization tables
# ──────────────────────────────────────────
NAME_ABBREVS = {
    # legal suffixes
    r'\bllc\b': '', r'\bllp\b': '', r'\binc\b': '', r'\bcorp\b': '',
    r'\bcorporation\b': '', r'\bltd\b': '', r'\blimited\b': '',
    r'\bpvt\b': '', r'\bprivate\b': '', r'\bco\b': '', r'\bcompany\b': '',
    r'\benterprises\b': '', r'\benterprise\b': '', r'\bgroup\b': '',
    r'\bassociates\b': '', r'\bservices\b': '', r'\bsolutions\b': '',
    r'\bindustries\b': '', r'\bindustry\b': '', r'\btraders\b': '',
    r'\btrading\b': '', r'\bholdings\b': '', r'\bholding\b': '',
    r'\bglobal\b': '', r'\binternational\b': '', r'\bnational\b': '',
    # punctuation shortcuts
    r'&': 'and', r'@': 'at',
}

ADDR_ABBREVS = {
    r'\bst\b': 'street', r'\bave\b': 'avenue', r'\brd\b': 'road',
    r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive', r'\bln\b': 'lane',
    r'\bct\b': 'court', r'\bpl\b': 'place', r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway', r'\bfwy\b': 'freeway', r'\bsq\b': 'square',
    r'\bapt\b': 'apartment', r'\bste\b': 'suite', r'\bflr\b': 'floor',
    r'\bfl\b': 'floor', r'\bno\b': 'number', r'\bn\b': 'north',
    r'\bs\b': 'south', r'\be\b': 'east', r'\bw\b': 'west',
}

PUNCT_TABLE = str.maketrans('', '', string.punctuation.replace('-', ''))
PAREN_RE    = re.compile(r'\(.*?\)')
MULTI_SPACE = re.compile(r'\s+')


def normalize_name(text: str) -> str:
    if not text:
        return ''
    t = text.lower()
    t = PAREN_RE.sub(' ', t)           # remove parenthetical notes
    t = t.translate(PUNCT_TABLE)       # strip punctuation
    for pat, rep in NAME_ABBREVS.items():
        t = re.sub(pat, rep, t)
    t = MULTI_SPACE.sub(' ', t).strip()
    return t


def normalize_addr(text: str) -> str:
    if not text:
        return ''
    t = text.lower()
    t = t.translate(PUNCT_TABLE)
    for pat, rep in ADDR_ABBREVS.items():
        t = re.sub(pat, rep, t)
    t = MULTI_SPACE.sub(' ', t).strip()
    return t


def trigrams(text: str):
    """Return the set of character trigrams for a string."""
    return set(text[i:i+3] for i in range(max(0, len(text)-2)))


def trigram_jaccard(tg1: set, tg2: set) -> float:
    if not tg1 or not tg2:
        return 0.0
    inter = len(tg1 & tg2)
    union = len(tg1 | tg2)
    return inter / union if union else 0.0


def token_jaccard(s1: str, s2: str) -> float:
    t1, t2 = set(s1.split()), set(s2.split())
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union else 0.0


def addr_token_overlap(a1: str, a2: str) -> float:
    """Jaccard on shared tokens, with extra weight on shared digit tokens."""
    t1, t2 = set(a1.split()), set(a2.split())
    if not t1 or not t2:
        return 0.0
    inter = t1 & t2
    union = t1 | t2
    if not union:
        return 0.0
    # Bonus: shared digits (street numbers, zip codes) are very informative
    digit_bonus = sum(1 for tok in inter if tok.isdigit()) * 0.1
    return min(1.0, len(inter) / len(union) + digit_bonus)


# ──────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────
def load_source(path: str) -> dict:
    """Load a source TSV into a dict {entity_id -> row_dict}."""
    records = {}
    t0 = time.time()
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            eid = row['entity_id']
            records[eid] = {
                'entity_id':    eid,
                'business_name': row.get('business_name', '') or '',
                'business_address': row.get('business_address', '') or '',
                'country':      row.get('country', '') or '',
                'norm_name':    normalize_name(row.get('business_name', '') or ''),
                'norm_addr':    normalize_addr(row.get('business_address', '') or ''),
            }
    print(f"  Loaded {len(records):,} records from {os.path.basename(path)} "
          f"in {time.time()-t0:.1f}s")
    return records


# ──────────────────────────────────────────
# Blocking: build a trigram index
# ──────────────────────────────────────────
def build_trigram_index(s23_data: dict) -> dict:
    """Build {trigram -> [entity_id, ...]} over all S2/S3 records."""
    t0 = time.time()
    index = collections.defaultdict(list)
    for eid, row in s23_data.items():
        tgs = trigrams(row['norm_name'])
        for tg in tgs:
            index[tg].append(eid)
    print(f"  Trigram index built: {len(index):,} trigrams "
          f"over {len(s23_data):,} records in {time.time()-t0:.1f}s")
    return index


def get_candidates(s1_row: dict, index: dict, s23_data: dict) -> list:
    """
    Retrieve and rank candidates for a single S1 record.
    Returns list of (cand_id, trigram_score) sorted by descending score.
    """
    name_tgs  = trigrams(s1_row['norm_name'])
    country   = s1_row['country']

    # Count trigram hits per candidate
    hit_counts = collections.Counter()
    for tg in name_tgs:
        if tg in index:
            for cid in index[tg]:
                hit_counts[cid] += 1

    if not hit_counts:
        return []

    # Convert raw hit count → approximate trigram Jaccard & filter by country
    max_hits = max(hit_counts.values()) if hit_counts else 1
    candidates = []
    for cid, hits in hit_counts.items():
        cand = s23_data.get(cid)
        if cand is None:
            continue
        # Country guard: must be same country (open-set friendly, just a string compare)
        if cand['country'] != country:
            continue
        # Rough Jaccard estimate: hits / (|tgs_s1| + |tgs_cand| - hits)
        cand_tgs_count = len(trigrams(cand['norm_name']))
        denom = len(name_tgs) + cand_tgs_count - hits
        rough_jac = hits / denom if denom > 0 else 0.0
        if rough_jac >= NAME_TRIGRAM_THRESHOLD:
            candidates.append((cid, rough_jac))

    # Keep top-K
    candidates.sort(key=lambda x: -x[1])
    return candidates[:TOP_K_CANDIDATES]


# ──────────────────────────────────────────
# Matching: score and threshold
# ──────────────────────────────────────────
def match_candidates(s1_row: dict, candidates: list, s23_data: dict):
    """
    Given the top-K candidates, compute fine-grained similarity scores
    and return (matched_ids, candidate_ids).
    """
    candidate_ids = []
    matched_ids   = []

    for cid, rough_jac in candidates:
        cand = s23_data[cid]
        candidate_ids.append(cid)

        # Fine name similarity: trigram Jaccard (precise)
        name_tg_jac = trigram_jaccard(
            trigrams(s1_row['norm_name']),
            trigrams(cand['norm_name'])
        )
        # Token-level Jaccard on name
        name_tok_jac = token_jaccard(s1_row['norm_name'], cand['norm_name'])

        # Best of trigram and token
        name_score = max(name_tg_jac, name_tok_jac)

        if name_score >= MATCH_THRESHOLD_HIGH:
            matched_ids.append(cid)
        elif name_score >= MATCH_THRESHOLD_LOW:
            # Fall back to address similarity
            addr_score = addr_token_overlap(s1_row['norm_addr'], cand['norm_addr'])
            if addr_score >= ADDR_THRESHOLD:
                matched_ids.append(cid)

    return matched_ids, candidate_ids


# ──────────────────────────────────────────
# Worker function (used by multiprocessing)
# ──────────────────────────────────────────
def _worker_init(shared_s23, shared_index):
    """Initialise per-process globals."""
    global G_S23, G_INDEX
    G_S23   = shared_s23
    G_INDEX = shared_index


def _worker_process_batch(batch):
    """Process a batch of S1 rows; returns list of (s1_id, matched, candidates)."""
    results = []
    for s1_row in batch:
        candidates = get_candidates(s1_row, G_INDEX, G_S23)
        matched, cands = match_candidates(s1_row, candidates, G_S23)
        results.append((s1_row['entity_id'], matched, cands))
    return results


# ──────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────
def run_pipeline(split: str = 'test'):
    """
    split: 'train' or 'test'
    """
    print(f"\n{'='*60}")
    print(f"  Amazon ML Challenge 2026 — Entity Resolution Pipeline")
    print(f"  Split: {split.upper()}   Workers: {N_WORKERS}")
    print(f"{'='*60}\n")

    data_dir = os.path.join(BASE, split)
    prefix   = 'train' if split == 'train' else 'test'

    print("Step 1: Loading data...")
    s1_data  = load_source(os.path.join(data_dir, f'{prefix}_source1.tsv'))
    s23_data = {}
    for src in ['2', '3']:
        part = load_source(os.path.join(data_dir, f'{prefix}_source{src}.tsv'))
        s23_data.update(part)

    print(f"\nStep 2: Building trigram index...")
    index = build_trigram_index(s23_data)

    print(f"\nStep 3: Processing {len(s1_data):,} Source-1 entities "
          f"with {N_WORKERS} workers...")

    # Split S1 records into batches for multiprocessing
    s1_rows  = list(s1_data.values())
    batch_sz = max(500, len(s1_rows) // (N_WORKERS * 4))
    batches  = [s1_rows[i:i+batch_sz] for i in range(0, len(s1_rows), batch_sz)]
    print(f"  Batch size: {batch_sz}  |  Total batches: {len(batches)}")

    all_results = []
    t0 = time.time()

    if N_WORKERS > 1:
        with mp.Pool(
            processes=N_WORKERS,
            initializer=_worker_init,
            initargs=(s23_data, index),
        ) as pool:
            done = 0
            for batch_results in pool.imap_unordered(_worker_process_batch, batches):
                all_results.extend(batch_results)
                done += len(batch_results)
                elapsed = time.time() - t0
                pct = 100 * done / len(s1_rows)
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (len(s1_rows) - done) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r  Progress: {done:,}/{len(s1_rows):,} "
                    f"({pct:.1f}%)  ETA: {eta:.0f}s   "
                )
                sys.stdout.flush()
    else:
        # Single-process fallback
        _worker_init(s23_data, index)
        for batch in batches:
            all_results.extend(_worker_process_batch(batch))

    print(f"\n  Finished in {time.time()-t0:.1f}s")

    print(f"\nStep 4: Writing output files to {OUT_DIR} ...")
    matching_path   = os.path.join(OUT_DIR, 'matching_results.tsv')
    candidates_path = os.path.join(OUT_DIR, 'candidate_pairs.tsv')

    with open(matching_path,   'w', encoding='utf-8', newline='') as mf, \
         open(candidates_path, 'w', encoding='utf-8', newline='') as cf:

        mw = csv.writer(mf, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        cw = csv.writer(cf, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')

        mw.writerow(['source1_entity_id', 'matched_entity_ids'])
        cw.writerow(['source1_entity_id', 'candidate_entity_ids'])

        # Sort by S1 entity ID for reproducibility
        all_results.sort(key=lambda x: x[0])

        stats_matched   = 0
        stats_singletons = 0
        stats_cands_total = 0

        for s1_id, matched, cands in all_results:
            m_str = ','.join(matched)
            c_str = ','.join(cands)
            mw.writerow([s1_id, m_str])
            cw.writerow([s1_id, c_str])

            if matched:
                stats_matched += 1
            else:
                stats_singletons += 1
            stats_cands_total += len(cands)

    print(f"\n{'='*60}")
    print(f"  OUTPUT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total S1 entities       : {len(all_results):,}")
    print(f"  Entities with matches   : {stats_matched:,}")
    print(f"  Singletons (no match)   : {stats_singletons:,}")
    avg_cands = stats_cands_total / len(all_results) if all_results else 0
    print(f"  Avg candidates per entity: {avg_cands:.2f}")
    print(f"\n  ✅  matching_results.tsv  → {matching_path}")
    print(f"  ✅  candidate_pairs.tsv   → {candidates_path}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────
# Local F_0.5 evaluation (train split only)
# ──────────────────────────────────────────
def evaluate_on_train():
    """
    Run pipeline on the TRAIN split, then score against ground truth.
    This is our local proxy for the leaderboard F_0.5 score.
    """
    run_pipeline(split='train')

    gt_path       = os.path.join(BASE, 'train', 'train_ground_truth.tsv')
    matching_path = os.path.join(OUT_DIR, 'matching_results.tsv')

    print("Evaluating against ground truth...")
    gt = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ids = row['matched_entity_ids'].strip()
            gt[row['source1_entity_id']] = set(ids.split(',')) if ids else set()

    preds = {}
    with open(matching_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ids = row['matched_entity_ids'].strip()
            preds[row['source1_entity_id']] = set(ids.split(',')) if ids else set()

    f_scores = []
    for s1_id, true_set in gt.items():
        pred_set = preds.get(s1_id, set())
        # Remove empty string artifact
        pred_set.discard('')
        true_set.discard('')

        if not pred_set and not true_set:
            f_scores.append(1.0)
            continue

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec == 0 and rec == 0:
            f_scores.append(0.0)
        else:
            f = (1.25 * prec * rec) / (0.25 * prec + rec)
            f_scores.append(f)

    macro_f05 = sum(f_scores) / len(f_scores) if f_scores else 0.0
    print(f"\n{'='*60}")
    print(f"  LOCAL F_0.5 SCORE: {macro_f05:.4f}")
    print(f"  (over {len(f_scores):,} Source-1 training entities)")
    print(f"{'='*60}\n")
    return macro_f05


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Amazon ML Challenge 2026 Pipeline')
    parser.add_argument('--mode', choices=['train', 'test', 'eval'], default='test',
                        help=('test=run on test split and write output, '
                              'train=run on train split (no scoring), '
                              'eval=run on train and score locally'))
    args = parser.parse_args()

    if args.mode == 'eval':
        evaluate_on_train()
    elif args.mode == 'train':
        run_pipeline(split='train')
    else:
        run_pipeline(split='test')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from rapidfuzz import fuzz
import argparse
import os
import shutil
import time

def find_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(current_dir, "../../..")),
        os.path.abspath(os.path.join(current_dir, "../..")),
        os.path.abspath(os.path.join(current_dir, "..")),
        os.getcwd()
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "student_resource/dataset")):
            return (
                os.path.join(candidate, "student_resource/dataset"),
                os.path.join(candidate, "output")
            )
    root = os.path.abspath(os.path.join(current_dir, "../../.."))
    return os.path.join(root, "student_resource/dataset"), os.path.join(root, "output")

def extract_pair_features(n1, a1, n2, a2):
    return [
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        fuzz.ratio(a1, a2) / 100.0,
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        fuzz.token_set_ratio(a1, a2) / 100.0,
        fuzz.partial_ratio(a1, a2) / 100.0,
    ]

def load_source_dicts(base_path, mode):
    print(f"Loading {mode} sources into fast in-memory lookup...")
    t0 = time.time()
    df1 = pd.read_csv(f"{base_path}/{mode}_source1.tsv", sep="\t").fillna('')
    df2 = pd.read_csv(f"{base_path}/{mode}_source2.tsv", sep="\t").fillna('')
    df3 = pd.read_csv(f"{base_path}/{mode}_source3.tsv", sep="\t").fillna('')
    
    names = {}
    addrs = {}
    
    for df in [df1, df2, df3]:
        for eid, name, addr in zip(df['entity_id'], df['business_name'], df['business_address']):
            names[eid] = str(name).lower()
            addrs[eid] = str(addr).lower()
            
    print(f"Loaded {len(names)} entities in {time.time()-t0:.2f}s")
    return df1['entity_id'].values, names, addrs

def train_matching_model(dataset_dir, output_dir, limit=None):
    train_base = os.path.join(dataset_dir, "train")
    _, names, addrs = load_source_dicts(train_base, "train")
    
    # Load ground truth
    gt_path = os.path.join(train_base, "train_ground_truth.tsv")
    true_matches = {}
    gt_df = pd.read_csv(gt_path, sep="\t")
    for _, row in gt_df.iterrows():
        val = row['matched_entity_ids']
        if pd.notna(val) and str(val).strip():
            true_matches[row['source1_entity_id']] = set(str(val).split(','))

    cand_path = os.path.join(output_dir, "candidate_pairs_train.tsv")
    print(f"Reading training candidates from {cand_path}...")
    
    X, y = [], []
    count = 0
    t0 = time.time()
    
    with open(cand_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            s1_id, tab, c_str = line.partition('\t')
            if not c_str:
                continue
            c_ids = c_str.split(',')
            n1 = names.get(s1_id, '')
            a1 = addrs.get(s1_id, '')
            
            s1_trues = true_matches.get(s1_id, set())
            for c_id in c_ids:
                n2 = names.get(c_id, '')
                a2 = addrs.get(c_id, '')
                feats = extract_pair_features(n1, a1, n2, a2)
                X.append(feats)
                y.append(1 if c_id in s1_trues else 0)
                
            count += 1
            if limit and count >= limit:
                break
                
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Extracted {len(X)} training pairs ({y.sum()} positive, {len(y)-y.sum()} negative) in {time.time()-t0:.2f}s")
    
    print("Fitting Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y)
    
    # Threshold tuning on training data for F_0.5
    probs = clf.predict_proba(X)[:, 1]
    best_thresh = 0.65
    best_f05 = 0.0
    for thresh in np.arange(0.50, 0.85, 0.05):
        preds = (probs >= thresh).astype(int)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f05 = fbeta_score(y, preds, beta=0.5, zero_division=0)
        print(f"  Thresh {thresh:.2f}: Precision={prec:.4f}, Recall={rec:.4f}, F_0.5={f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"Selected optimal confidence threshold: {best_thresh:.2f} (F_0.5: {best_f05:.4f})")
    return clf, best_thresh

def generate_test_predictions(clf, threshold, dataset_dir, output_dir, limit=None):
    test_base = os.path.join(dataset_dir, "test")
    test_s1_ids, names, addrs = load_source_dicts(test_base, "test")
    
    cand_path = os.path.join(output_dir, "candidate_pairs_test.tsv")
    results_path = os.path.join(output_dir, "matching_results.tsv")
    canonical_cand = os.path.join(output_dir, "candidate_pairs.tsv")
    
    shutil.copy(cand_path, canonical_cand)
    print(f"Copied candidate_pairs_test.tsv to {canonical_cand}")
    
    print(f"Generating test predictions to {results_path}...")
    t0 = time.time()
    
    # We will process candidates and predict
    total_processed = 0
    total_matches = 0
    
    batch_features = []
    batch_meta = []  # (s1_id, c_id)
    s1_matches = {s1: [] for s1 in test_s1_ids}
    
    def flush_batch():
        nonlocal batch_features, batch_meta, total_matches
        if not batch_features:
            return
        X_batch = np.array(batch_features, dtype=np.float32)
        probs = clf.predict_proba(X_batch)[:, 1]
        for (s1_id, c_id), prob in zip(batch_meta, probs):
            if prob >= threshold:
                s1_matches[s1_id].append(c_id)
                total_matches += 1
        batch_features = []
        batch_meta = []
    
    with open(cand_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            s1_id, tab, c_str = line.partition('\t')
            if not c_str:
                continue
            c_ids = c_str.split(',')
            n1 = names.get(s1_id, '')
            a1 = addrs.get(s1_id, '')
            
            for c_id in c_ids:
                n2 = names.get(c_id, '')
                a2 = addrs.get(c_id, '')
                batch_features.append(extract_pair_features(n1, a1, n2, a2))
                batch_meta.append((s1_id, c_id))
                
            total_processed += 1
            if len(batch_features) >= 100000:
                flush_batch()
                print(f"  Evaluated {total_processed} S1 entities ({total_matches} positive matches)...")
                
    flush_batch()
    
    # Write matching_results.tsv in the exact order of test_s1_ids
    print(f"Writing final matches for {len(test_s1_ids)} S1 entities...")
    with open(results_path, 'w', encoding='utf-8') as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1_ids:
            matches = s1_matches.get(s1_id, [])
            if matches:
                # Deduplicate preserving order
                seen = set()
                deduped = [x for x in matches if not (x in seen or seen.add(x))]
                out_f.write(f"{s1_id}\t{','.join(deduped)}\n")
            else:
                out_f.write(f"{s1_id}\t\n")
                
    print(f"Saved matching results to {results_path} in {time.time()-t0:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit for training pairs')
    args = parser.parse_args()
    
    dataset_dir, output_dir = find_paths()
    clf, best_thresh = train_matching_model(dataset_dir, output_dir, limit=args.limit)
    generate_test_predictions(clf, best_thresh, dataset_dir, output_dir, limit=args.limit)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn
import time
import os
import shutil
import argparse
from collections import defaultdict
import gc

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
    # Default fallback
    root = os.path.abspath(os.path.join(current_dir, "../../.."))
    return os.path.join(root, "student_resource/dataset"), os.path.join(root, "output")

def create_blocking(mode='train', limit=None, top_n=6, threshold=0.20):
    dataset_dir, output_dir = find_paths()
    base_path = os.path.join(dataset_dir, mode)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {mode} datasets from {base_path}...")
    df1 = pd.read_csv(f"{base_path}/{mode}_source1.tsv", sep="\t").fillna('')
    df2 = pd.read_csv(f"{base_path}/{mode}_source2.tsv", sep="\t").fillna('')
    df3 = pd.read_csv(f"{base_path}/{mode}_source3.tsv", sep="\t").fillna('')

    print(f"Total records - S1: {len(df1)}, S2: {len(df2)}, S3: {len(df3)}")

    # For training, if limit is not set, default to 4000 S1 records to keep training fast & diverse
    if mode == 'train' and limit is None:
        limit = 4000

    if limit is not None:
        print(f"Limiting S1 active candidate search to first {limit} records...")
        df1_search = df1.head(limit).copy()
    else:
        df1_search = df1

    # Clean and combine text
    print("Normalizing text for TF-IDF...")
    for df in [df1_search, df2, df3]:
        df['combined_text'] = (df['business_name'].astype(str) + " " + df['business_address'].astype(str)).str.lower()

    countries = [c for c in df1_search['country'].unique() if c]
    results = defaultdict(list)
    batch_size = 50000

    for country in countries:
        print(f"\n{'='*40}\nProcessing country: {country}")
        c_df1 = df1_search[df1_search['country'] == country].reset_index(drop=True)
        c_df2 = df2[df2['country'] == country].reset_index(drop=True)
        c_df3 = df3[df3['country'] == country].reset_index(drop=True)
        
        print(f"  S1: {len(c_df1)} | S2: {len(c_df2)} | S3: {len(c_df3)}")
        if len(c_df1) == 0:
            continue
            
        print("  Fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            min_df=2,
            max_features=80000,
            sublinear_tf=True
        )
        
        # Fit on sample of S2 and S3 for representative vocabulary
        sample_corpus = []
        if len(c_df2) > 0:
            sample_corpus.extend(c_df2['combined_text'].sample(min(80000, len(c_df2)), random_state=42))
        if len(c_df3) > 0:
            sample_corpus.extend(c_df3['combined_text'].sample(min(80000, len(c_df3)), random_state=42))
        if len(sample_corpus) == 0:
            sample_corpus = c_df1['combined_text']
            
        vectorizer.fit(sample_corpus)
        
        print("  Transforming S2 and S3...")
        t0 = time.time()
        M2 = vectorizer.transform(c_df2['combined_text']) if len(c_df2) > 0 else None
        M3 = vectorizer.transform(c_df3['combined_text']) if len(c_df3) > 0 else None
        M2_T = M2.transpose() if M2 is not None else None
        M3_T = M3.transpose() if M3 is not None else None
        print(f"  Transformed S2 & S3 in {time.time()-t0:.2f}s")
        
        c_s1_ids = c_df1['entity_id'].to_numpy()
        c_s2_ids = c_df2['entity_id'].to_numpy() if len(c_df2) > 0 else np.array([])
        c_s3_ids = c_df3['entity_id'].to_numpy() if len(c_df3) > 0 else np.array([])
        
        num_batches = int(np.ceil(len(c_df1) / batch_size))
        print(f"  Processing S1 in {num_batches} batches...")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(c_df1))
            batch_texts = c_df1['combined_text'].iloc[start_idx:end_idx]
            batch_s1_ids = c_s1_ids[start_idx:end_idx]
            
            M1_batch = vectorizer.transform(batch_texts)
            
            if M2_T is not None:
                matches_s2 = sparse_dot_topn.sp_matmul_topn(
                    M1_batch, M2_T, top_n=top_n, threshold=threshold, n_threads=-1
                )
                rows, cols = matches_s2.nonzero()
                for r, c in zip(rows, cols):
                    results[batch_s1_ids[r]].append(c_s2_ids[c])
                    
            if M3_T is not None:
                matches_s3 = sparse_dot_topn.sp_matmul_topn(
                    M1_batch, M3_T, top_n=top_n, threshold=threshold, n_threads=-1
                )
                rows, cols = matches_s3.nonzero()
                for r, c in zip(rows, cols):
                    results[batch_s1_ids[r]].append(c_s3_ids[c])
                    
            if (i + 1) % 5 == 0 or (i + 1) == num_batches:
                print(f"    Batch {i+1}/{num_batches} done.")
                
        del M2, M3, M2_T, M3_T, c_df2, c_df3, vectorizer
        gc.collect()

    out_path = os.path.join(output_dir, f"candidate_pairs_{mode}.tsv")
    print(f"\nSaving candidates to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        # Ensure EVERY single S1 entity from df1 is present
        for s1_id in df1['entity_id'].values:
            candidates = results.get(s1_id, [])
            if candidates:
                seen = set()
                deduped = [x for x in candidates if not (x in seen or seen.add(x))]
                f.write(f"{s1_id}\t{','.join(deduped)}\n")
            else:
                f.write(f"{s1_id}\t\n")
                
    print(f"Successfully saved to {out_path}")
    
    if mode == 'test':
        canonical_cand = os.path.join(output_dir, "candidate_pairs.tsv")
        shutil.copy(out_path, canonical_cand)
        print(f"Copied to {canonical_cand}")

    if mode == 'train':
        print("\nEvaluating Blocking Recall on processed subset...")
        gt_path = os.path.join(base_path, "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            gt_df = pd.read_csv(gt_path, sep="\t")
            processed_s1_ids = set(df1_search['entity_id'].values)
            gt_df = gt_df[gt_df['source1_entity_id'].isin(processed_s1_ids)]
            
            total_true = 0
            total_found = 0
            for _, row in gt_df.iterrows():
                val = row['matched_entity_ids']
                if pd.isna(val) or not str(val).strip():
                    continue
                true_ids = set(str(val).split(','))
                s1_id = row['source1_entity_id']
                cand_ids = set(results.get(s1_id, []))
                total_true += len(true_ids)
                total_found += len(true_ids.intersection(cand_ids))
                
            if total_true > 0:
                recall = total_found / total_true
                print(f"Candidate Recall: {recall:.4f} ({total_found} / {total_true})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--limit', type=int, default=None, help='Number of S1 rows to actively search')
    parser.add_argument('--top_n', type=int, default=6, help='Top N matches per source')
    parser.add_argument('--threshold', type=float, default=0.20, help='Similarity threshold')
    args = parser.parse_args()
    create_blocking(mode=args.mode, limit=args.limit, top_n=args.top_n, threshold=args.threshold)

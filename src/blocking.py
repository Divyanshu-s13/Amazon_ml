import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn
import time
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import gc

def create_blocking(limit=None):
    base_path = '/Users/divyanshusingh/Desktop/ML_challenge/student_resource/dataset/train'
    
    print("Loading data...")
    # Load all data, but only process 'limit' S1 rows if specified
    df1 = pd.read_csv(f"{base_path}/train_source1.tsv", sep="\t").fillna('')
    if limit:
        df1 = df1.head(limit)
    
    df2 = pd.read_csv(f"{base_path}/train_source2.tsv", sep="\t").fillna('')
    df3 = pd.read_csv(f"{base_path}/train_source3.tsv", sep="\t").fillna('')

    # Clean text
    print("Cleaning text...")
    for df in [df1, df2, df3]:
        df['combined_text'] = (df['business_name'].astype(str) + " " + df['business_address'].astype(str)).str.lower()
        # Basic normalization could be added here

    countries = set(df1['country'].unique())
    countries = [c for c in countries if c]

    results = defaultdict(list)
    top_n = 20
    batch_size = 50000

    for country in countries:
        print(f"\n{'='*40}\nProcessing country: {country}")
        
        c_df1 = df1[df1['country'] == country].reset_index(drop=True)
        c_df2 = df2[df2['country'] == country].reset_index(drop=True)
        c_df3 = df3[df3['country'] == country].reset_index(drop=True)
        
        print(f"  S1: {len(c_df1)} | S2: {len(c_df2)} | S3: {len(c_df3)}")
        
        if len(c_df1) == 0: continue
            
        print("  Fitting TF-IDF on a sample...")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=150000)
        
        # Fit on a sample to save time/memory
        sample_size = min(200000, len(c_df2))
        sample_corpus = c_df2['combined_text'].sample(n=sample_size, random_state=42) if len(c_df2) > 0 else c_df1['combined_text']
        vectorizer.fit(sample_corpus)
        
        print("  Transforming S2 and S3...")
        start = time.time()
        M2 = vectorizer.transform(c_df2['combined_text']) if len(c_df2) > 0 else None
        M3 = vectorizer.transform(c_df3['combined_text']) if len(c_df3) > 0 else None
        print(f"  Transformed in {time.time()-start:.2f}s")
        
        M2_T = M2.transpose() if M2 is not None else None
        M3_T = M3.transpose() if M3 is not None else None
        
        # Process S1 in batches to save memory
        num_batches = int(np.ceil(len(c_df1) / batch_size))
        print(f"  Processing S1 in {num_batches} batches...")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(c_df1))
            batch_df = c_df1.iloc[start_idx:end_idx]
            
            M1_batch = vectorizer.transform(batch_df['combined_text'])
            
            if M2_T is not None:
                matches_s2 = sparse_dot_topn.sp_matmul_topn(M1_batch, M2_T, top_n=top_n)
                rows, cols = matches_s2.nonzero()
                for r, c in zip(rows, cols):
                    s1_id = batch_df.iloc[r]['entity_id']
                    s2_id = c_df2.iloc[c]['entity_id']
                    results[s1_id].append(s2_id)
                    
            if M3_T is not None:
                matches_s3 = sparse_dot_topn.sp_matmul_topn(M1_batch, M3_T, top_n=top_n)
                rows, cols = matches_s3.nonzero()
                for r, c in zip(rows, cols):
                    s1_id = batch_df.iloc[r]['entity_id']
                    s3_id = c_df3.iloc[c]['entity_id']
                    results[s1_id].append(s3_id)
                    
            print(f"    Batch {i+1}/{num_batches} done.")
            
        # Free up memory
        del M2, M3, M2_T, M3_T, c_df2, c_df3, vectorizer
        gc.collect()

    print("\nSaving candidate_pairs.tsv...")
    out_dir = '/Users/divyanshusingh/Desktop/ML_challenge/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/candidate_pairs.tsv"
    
    with open(out_path, 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in df1['entity_id'].values:
            candidates = results.get(s1_id, [])
            candidates = sorted(list(set(candidates)))
            f.write(f"{s1_id}\t{','.join(candidates)}\n")
            
    print(f"Successfully saved to {out_path}")
    
    if limit:
        print("\nEvaluating Blocking Recall on subset...")
        gt_df = pd.read_csv(f"{base_path}/train_ground_truth.tsv", sep="\t")
        
        # Only evaluate on S1 ids that we processed
        processed_s1_ids = set(df1['entity_id'].values)
        gt_df = gt_df[gt_df['source1_entity_id'].isin(processed_s1_ids)]
        
        total_true_matches = 0
        total_found_matches = 0
        
        for _, row in gt_df.iterrows():
            if pd.isna(row['matched_entity_ids']) or not str(row['matched_entity_ids']).strip():
                continue
            
            true_matches = set(row['matched_entity_ids'].split(','))
            s1_id = row['source1_entity_id']
            
            total_true_matches += len(true_matches)
            found_candidates = set(results.get(s1_id, []))
            total_found_matches += len(true_matches.intersection(found_candidates))
            
        if total_true_matches > 0:
            recall = total_found_matches / total_true_matches
            print(f"Recall: {recall:.4f} ({total_found_matches} / {total_true_matches})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Number of S1 rows to process')
    args = parser.parse_args()
    create_blocking(limit=args.limit)

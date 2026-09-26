import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
import argparse
import os
import time

def extract_features(s1_row, s2_row):
    """Compute string similarity features between two records."""
    name1, name2 = str(s1_row.get('business_name', '')), str(s2_row.get('business_name', ''))
    addr1, addr2 = str(s1_row.get('business_address', '')), str(s2_row.get('business_address', ''))
    
    name1 = name1.lower()
    name2 = name2.lower()
    addr1 = addr1.lower()
    addr2 = addr2.lower()

    # RapidFuzz gives ratios between 0 and 100
    features = [
        fuzz.ratio(name1, name2) / 100.0,
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        fuzz.token_set_ratio(name1, name2) / 100.0,
        fuzz.ratio(addr1, addr2) / 100.0,
        fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        fuzz.token_set_ratio(addr1, addr2) / 100.0
    ]
    return features

def build_training_data(limit=None):
    base_path = '/Users/divyanshusingh/Desktop/ML_challenge/student_resource/dataset/train'
    output_dir = '/Users/divyanshusingh/Desktop/ML_challenge/output'
    
    print("Loading datasets...")
    df1 = pd.read_csv(f"{base_path}/train_source1.tsv", sep="\t", index_col='entity_id')
    df2 = pd.read_csv(f"{base_path}/train_source2.tsv", sep="\t", index_col='entity_id')
    df3 = pd.read_csv(f"{base_path}/train_source3.tsv", sep="\t", index_col='entity_id')
    
    gt_df = pd.read_csv(f"{base_path}/train_ground_truth.tsv", sep="\t")
    
    # Create quick lookup for ground truth
    print("Building ground truth index...")
    true_matches = {}
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids'])
        if matches and matches.lower() != 'nan':
            true_matches[row['source1_entity_id']] = set(matches.split(','))
        else:
            true_matches[row['source1_entity_id']] = set()

    print("Loading candidate pairs...")
    candidates_df = pd.read_csv(f"{output_dir}/candidate_pairs.tsv", sep="\t")
    
    if limit:
        candidates_df = candidates_df.head(limit)
        
    X = []
    y = []
    metadata = [] # To keep track of (s1_id, candidate_id)
    
    print("Extracting features (this might take a while)...")
    start = time.time()
    for _, row in candidates_df.iterrows():
        s1_id = row['source1_entity_id']
        c_str = str(row['candidate_entity_ids'])
        if not c_str or c_str.lower() == 'nan':
            continue
            
        c_ids = c_str.split(',')
        s1_row = df1.loc[s1_id]
        
        for c_id in c_ids:
            if c_id.startswith('S2'):
                c_row = df2.loc[c_id]
            else:
                c_row = df3.loc[c_id]
                
            features = extract_features(s1_row, c_row)
            is_match = 1 if c_id in true_matches.get(s1_id, set()) else 0
            
            X.append(features)
            y.append(is_match)
            metadata.append((s1_id, c_id))
            
    print(f"Feature extraction took {time.time()-start:.2f}s")
    return np.array(X), np.array(y), metadata

def train_and_evaluate(X, y):
    print("Training XGBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\n--- Evaluation on Test Split ---")
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F_0.5:     {f05:.4f}")
    
    return clf, X, y, metadata

def generate_submission(clf, metadata, X, output_dir):
    print("\nGenerating matching_results.tsv...")
    # Predict probabilities
    y_probs = clf.predict_proba(X)[:, 1]
    
    # We want high precision, so we use a strict threshold
    threshold = 0.65
    y_pred = (y_probs >= threshold).astype(int)
    
    # Group predictions by Source 1 entity
    results = {}
    for i, (s1_id, c_id) in enumerate(metadata):
        if s1_id not in results:
            results[s1_id] = []
        if y_pred[i] == 1:
            results[s1_id].append(c_id)
            
    out_path = f"{output_dir}/matching_results.tsv"
    with open(out_path, 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in results.items():
            f.write(f"{s1_id}\t{','.join(matches)}\n")
            
    print(f"Submission saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Number of candidate rows to process')
    args = parser.parse_args()
    
    output_dir = '/Users/divyanshusingh/Desktop/ML_challenge/output'
    X, y, metadata = build_training_data(limit=args.limit)
    if len(X) > 0:
        model, _, _, _ = train_and_evaluate(X, y)
        generate_submission(model, metadata, X, output_dir)
    else:
        print("No candidates to process.")

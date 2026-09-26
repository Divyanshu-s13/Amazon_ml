import csv
import collections
import string
import time
from difflib import SequenceMatcher

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def string_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()

def jaccard(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def run_prototype():
    base_path = '/Users/divyanshusingh/Desktop/ML_challenge/student_resource/dataset/train'
    
    # 1. Load Data
    s1_data = {}
    s23_data = {}
    
    limit = 100000
    
    print("Loading data...")
    for file, target_dict in [('train_source1.tsv', s1_data), ('train_source2.tsv', s23_data), ('train_source3.tsv', s23_data)]:
        with open(f"{base_path}/{file}", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i >= limit: break
                row['norm_name'] = normalize_text(row.get('business_name', ''))
                row['norm_address'] = normalize_text(row.get('business_address', ''))
                target_dict[row['entity_id']] = row

    # 2. Build Inverted Index (Blocking)
    print("Building Inverted Index on Source 2 and 3...")
    start = time.time()
    inverted_index = collections.defaultdict(list)
    for eid, row in s23_data.items():
        words = set(row['norm_name'].split())
        for w in words:
            if len(w) > 2: # Skip very short words for index
                inverted_index[w].append(eid)
    print(f"Index built in {time.time()-start:.2f}s. Vocab size: {len(inverted_index)}")

    # 3. Ground Truth
    gt_matches = collections.defaultdict(set)
    with open(f"{base_path}/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= limit: break
            if row.get('matched_entity_ids'):
                gt_matches[row['source1_entity_id']] = set(row['matched_entity_ids'].split(','))

    # 4. Generate Candidates & Match
    print("Generating candidates and matching...")
    start = time.time()
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # We will compute F_0.5 per S1 entity
    entity_f_scores = []
    
    for i, (s1_id, s1_row) in enumerate(s1_data.items()):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i} Source 1 records...")
            
        words = set(s1_row['norm_name'].split())
        candidate_counts = collections.defaultdict(int)
        
        for w in words:
            if len(w) > 2 and w in inverted_index:
                for eid in inverted_index[w]:
                    candidate_counts[eid] += 1
                    
        # Sort candidates by overlap count
        top_candidates = sorted(candidate_counts.items(), key=lambda x: -x[1])[:15]
        
        predicted_matches = []
        for cand_id, count in top_candidates:
            cand_row = s23_data[cand_id]
            # Must match country
            if s1_row['country'] != cand_row['country']:
                continue
                
            name_sim = string_similarity(s1_row['norm_name'], cand_row['norm_name'])
            addr_sim = string_similarity(s1_row['norm_address'], cand_row['norm_address'])
            
            # Simple heuristic threshold
            if name_sim > 0.75 or (name_sim > 0.6 and addr_sim > 0.6):
                predicted_matches.append(cand_id)
                
        pred_set = set(predicted_matches)
        true_set = gt_matches.get(s1_id, set())
        
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        if len(pred_set) == 0 and len(true_set) == 0:
            f_score = 1.0 # Correct singleton
        elif len(pred_set) == 0 or len(true_set) == 0:
            f_score = 0.0 # False positive singleton or completely missed
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision == 0 and recall == 0:
                f_score = 0.0
            else:
                f_score = (1.25 * precision * recall) / (0.25 * precision + recall)
                
        entity_f_scores.append(f_score)

    print(f"Inference completed in {time.time()-start:.2f}s")
    print(f"Macro-Averaged F_0.5 Score: {sum(entity_f_scores)/len(entity_f_scores):.4f}")

if __name__ == '__main__':
    run_prototype()

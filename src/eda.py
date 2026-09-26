import csv
import collections
import time

def explore_data():
    base_path = '/Users/divyanshusingh/Desktop/ML_challenge/student_resource/dataset/train'
    
    def analyze_file(filename, limit=100000):
        print(f"\n--- {filename} ---")
        row_count = 0
        missing = collections.defaultdict(int)
        country_dist = collections.defaultdict(int)
        
        start = time.time()
        with open(f"{base_path}/{filename}", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row_count += 1
                for k, v in row.items():
                    if not v or v.strip() == "":
                        missing[k] += 1
                
                if "country" in row:
                    country_dist[row["country"]] += 1
                
                if row_count >= limit:
                    break
        
        print(f"Read {row_count} rows in {time.time()-start:.2f}s")
        print("Missing values:")
        for k, v in missing.items():
            print(f"  {k}: {v}")
            
        if country_dist:
            print("Country distribution:")
            for k, v in sorted(country_dist.items(), key=lambda x: -x[1]):
                print(f"  {k}: {v}")

    analyze_file('train_source1.tsv')
    analyze_file('train_source2.tsv')
    analyze_file('train_source3.tsv')

    print("\n--- Ground Truth (Subset) ---")
    row_count = 0
    has_match_count = 0
    match_lengths = []
    
    with open(f"{base_path}/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row_count += 1
            matches = row.get("matched_entity_ids", "")
            if matches and matches.strip():
                has_match_count += 1
                match_lengths.append(len(matches.split(",")))
                
            if row_count >= 100000:
                break
                
    print(f"Total labeled entities (in subset): {row_count}")
    print(f"Entities with at least 1 match: {has_match_count}")
    print(f"Singletons (no match): {row_count - has_match_count}")
    if match_lengths:
        print(f"Avg matches per entity (when matched): {sum(match_lengths)/len(match_lengths):.2f}")
        print(f"Max matches per entity: {max(match_lengths)}")

if __name__ == '__main__':
    explore_data()

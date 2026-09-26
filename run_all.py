import os
import subprocess
import argparse

def run(cmd):
    print(f"\n--- Running: {cmd} ---")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000, help='Limit for quick testing')
    args = parser.parse_args()
    
    limit_flag = f"--limit {args.limit}" if args.limit else ""
    
    print("Step 1: Generating candidates for TRAIN set (for model training)")
    run(f"python3 src/blocking.py --mode train {limit_flag}")
    
    print("\nStep 2: Generating candidates for TEST set (for final predictions)")
    run(f"python3 src/blocking.py --mode test {limit_flag}")
    
    print("\nStep 3: Training model and generating submission")
    run(f"python3 src/pipeline.py {limit_flag}")
    
    print("\nStep 4: Validating submission")
    run("python3 ../student_resource/utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir ../student_resource/dataset/test")
    
    print("\nAll done! Submission files are ready in the output/ directory.")

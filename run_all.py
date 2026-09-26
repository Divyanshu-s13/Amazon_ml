import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Locate project root
PROJECT_ROOT = None
for candidate in [
    os.path.abspath(os.path.join(SCRIPT_DIR, "../..")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
    os.getcwd()
]:
    if os.path.exists(os.path.join(candidate, "student_resource/utils/validate_submission.py")):
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

PYTHON = sys.executable

def run(cmd, cwd=None):
    print(f"\n=======================================================")
    print(f"RUNNING: {cmd}")
    print(f"=======================================================")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd or SCRIPT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument('--limit', type=int, default=None, help='Limit S1 active search for quick testing (e.g. 1000)')
    parser.add_argument('--skip-train-blocking', action='store_true', help='Skip step 1 if train candidates already exist')
    parser.add_argument('--package', action='store_true', default=True, help='Automatically package into submission zip')
    args = parser.parse_args()

    limit_flag = f"--limit {args.limit}" if args.limit else ""
    
    output_dir = os.path.join(PROJECT_ROOT, "output")
    student_resource_dir = os.path.join(PROJECT_ROOT, "student_resource")
    validator = os.path.join(student_resource_dir, "utils/validate_submission.py")
    test_dir = os.path.join(student_resource_dir, "dataset/test")
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    # Step 1: Train candidate generation
    train_cand = os.path.join(output_dir, "candidate_pairs_train.tsv")
    if args.skip_train_blocking and os.path.exists(train_cand):
        print("\nStep 1: Skipping train candidate generation (already exists)")
    else:
        print("\nStep 1: Generating candidates for TRAIN set (for model training)")
        run(f'"{PYTHON}" src/blocking.py --mode train {limit_flag}')

    # Step 2: Test candidate generation
    print("\nStep 2: Generating candidates for TEST set (for final predictions)")
    run(f'"{PYTHON}" src/blocking.py --mode test {limit_flag}')

    # Step 3: Model training and test inference
    print("\nStep 3: Training matching model and generating predictions")
    run(f'"{PYTHON}" src/pipeline.py {limit_flag}')

    # Step 4: Submission validation
    print("\nStep 4: Validating submission files against official constraints")
    run(f'"{PYTHON}" "{validator}" --matching "{matching_tsv}" --candidate "{candidate_tsv}" --test-dir "{test_dir}"')

    # Step 5: Packaging
    if args.package:
        package_script = os.path.join(PROJECT_ROOT, "package.sh")
        if os.path.exists(package_script):
            print("\nStep 5: Packaging into submission ZIP")
            run(f'bash "{package_script}"', cwd=PROJECT_ROOT)

    print("\n=======================================================")
    print("ALL DONE! Pipeline completed successfully.")
    print(f"Submission zip: {os.path.join(PROJECT_ROOT, 'Divyanshu_submission.zip')}")
    print("=======================================================\n")

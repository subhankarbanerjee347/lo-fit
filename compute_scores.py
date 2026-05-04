"""
Compute MC1/MC2 scores from answer_dump CSVs.
Usage: python compute_scores.py
Reads all CSVs in ./tqa_results/answer_dump/ and prints MC1/MC2 for each.
"""
import pandas as pd
import numpy as np
import os
import glob

def compute_mc1(row, model_key):
    """MC1: Is the best (highest prob) answer the correct one?"""
    col = f'{model_key} MC1'
    if col in row.index and not pd.isna(row[col]):
        return row[col]
    return np.nan

def compute_mc2(row, model_key):
    """MC2: Normalized probability mass on correct answers."""
    col = f'{model_key} MC2'
    if col in row.index and not pd.isna(row[col]):
        return row[col]
    return np.nan

def extract_scores(csv_path):
    """Extract MC1 and MC2 from an answer_dump CSV."""
    df = pd.read_csv(csv_path)

    # Find model key from column names (look for "XXX MC1" columns)
    mc1_cols = [c for c in df.columns if c.endswith(' MC1')]
    mc2_cols = [c for c in df.columns if c.endswith(' MC2')]

    results = {}
    for mc1_col in mc1_cols:
        model_key = mc1_col.replace(' MC1', '')
        mc2_col = f'{model_key} MC2'

        mc1_score = df[mc1_col].mean()
        mc2_score = df[mc2_col].mean() if mc2_col in df.columns else np.nan
        results[model_key] = {'MC1': mc1_score, 'MC2': mc2_score}

    return results

if __name__ == '__main__':
    dump_dir = './tqa_results/answer_dump'
    csv_files = sorted(glob.glob(os.path.join(dump_dir, '*.csv')))

    if not csv_files:
        print(f"No CSV files found in {dump_dir}")
        exit(1)

    print("=" * 60)
    print("TruthfulQA MC1/MC2 Results")
    print("=" * 60)

    # Collect results for averaging across folds
    all_results = {}

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        print(f"\n--- {fname} ---")

        results = extract_scores(csv_path)
        for model_key, scores in results.items():
            print(f"  {model_key}: MC1 = {scores['MC1']:.4f}, MC2 = {scores['MC2']:.4f}")

            # Group by variant (strip fold info for averaging)
            # e.g., "llama2_7B" from different fold files
            if model_key not in all_results:
                all_results[model_key] = {'MC1': [], 'MC2': []}
            all_results[model_key]['MC1'].append(scores['MC1'])
            all_results[model_key]['MC2'].append(scores['MC2'])

    # Print summary grouped by variant name from filename
    print("\n" + "=" * 60)
    print("Summary (per file)")
    print("=" * 60)

    # Group files by variant (strip fold number from filename)
    variant_scores = {}
    for csv_path in csv_files:
        fname = os.path.basename(csv_path).replace('.csv', '')
        results = extract_scores(csv_path)

        # Determine variant from filename
        for variant in ['v_rope', 'rope', '_v_']:
            if variant in fname:
                variant_key = variant.strip('_')
                break
        else:
            variant_key = fname

        if variant_key not in variant_scores:
            variant_scores[variant_key] = {'MC1': [], 'MC2': []}

        for model_key, scores in results.items():
            variant_scores[variant_key]['MC1'].append(scores['MC1'])
            variant_scores[variant_key]['MC2'].append(scores['MC2'])

    print(f"\n{'Variant':<15} {'MC1 (avg)':<12} {'MC2 (avg)':<12}")
    print("-" * 40)
    for variant, scores in sorted(variant_scores.items()):
        mc1_avg = np.mean(scores['MC1'])
        mc2_avg = np.mean(scores['MC2'])
        print(f"{variant:<15} {mc1_avg:<12.4f} {mc2_avg:<12.4f}")

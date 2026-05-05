"""
Compute scores across all tasks and models.
Usage: python compute_scores.py

Reads results from:
  - TruthfulQA: ./tqa_results/answer_dump/*.csv  (MC1/MC2)
  - MQuAKE:     ./finetuned_outputs/mquake/*/outputs.json  (Exact Match)
  - CLUTRR:     ./finetuned_outputs/clutrr/*/outputs.json   (Exact Match)
"""
import pandas as pd
import numpy as np
import os
import json
import glob
import re


def compute_tqa_scores():
    """Compute MC1/MC2 from TruthfulQA answer_dump CSVs."""
    dump_dir = './tqa_results/answer_dump'
    csv_files = sorted(glob.glob(os.path.join(dump_dir, '*.csv')))

    if not csv_files:
        return None

    results = []
    for csv_path in csv_files:
        fname = os.path.basename(csv_path).replace('.csv', '')
        df = pd.read_csv(csv_path)

        mc1_cols = [c for c in df.columns if c.endswith(' MC1')]
        for mc1_col in mc1_cols:
            model_key = mc1_col.replace(' MC1', '')
            mc2_col = f'{model_key} MC2'

            mc1 = df[mc1_col].mean()
            mc2 = df[mc2_col].mean() if mc2_col in df.columns else np.nan

            # Extract variant from filename
            variant = extract_variant(fname)

            results.append({
                'task': 'TruthfulQA',
                'model': model_key,
                'variant': variant,
                'file': fname,
                'metric': 'MC1',
                'score': mc1,
            })
            results.append({
                'task': 'TruthfulQA',
                'model': model_key,
                'variant': variant,
                'file': fname,
                'metric': 'MC2',
                'score': mc2,
            })

    return results


def compute_json_scores(task_name):
    """Compute Exact Match accuracy from outputs.json files."""
    # Search both locations:
    #   1) Root-level dirs like llama2_7B_mquake_rope_seed42/outputs.json
    #   2) Nested dirs like ./finetuned_outputs/mquake/*/outputs.json
    output_dirs = sorted(
        glob.glob(f'./*_{task_name}_*/outputs.json') +
        glob.glob(f'./finetuned_outputs/{task_name}/*/outputs.json')
    )

    if not output_dirs:
        return None

    results = []
    for json_path in output_dirs:
        dir_name = os.path.basename(os.path.dirname(json_path))
        data = json.load(open(json_path))

        if len(data) == 0:
            continue

        acc = sum(d['correct'] for d in data) / len(data)
        total = len(data)
        correct = sum(d['correct'] for d in data)

        # Extract model and variant from directory name
        variant = extract_variant(dir_name)
        model = extract_model(dir_name)

        results.append({
            'task': task_name.upper(),
            'model': model,
            'variant': variant,
            'file': dir_name,
            'metric': 'EM',
            'score': acc,
            'correct': correct,
            'total': total,
        })

    return results


def extract_variant(name):
    """Extract variant (v_rope, rope, v) from a filename/dirname."""
    # Order matters: check v_rope before rope and v
    if 'v_rope' in name:
        return 'v_rope'
    elif '_rope' in name or name.startswith('rope'):
        return 'rope'
    elif '_v_' in name or name.endswith('_v') or 'lofit' in name:
        return 'v'
    elif 'Aonly' in name:
        return 'A (head selection)'
    else:
        return name


def extract_model(name):
    """Extract model name from a filename/dirname."""
    if 'llama2_13B' in name:
        return 'llama2_13B'
    elif 'llama2_7B' in name:
        return 'llama2_7B'
    elif 'gemma_7b' in name:
        return 'gemma_7b'
    else:
        return 'unknown'


def print_task_results(task_name, results):
    """Print results for a single task, grouped by model and variant."""
    if not results:
        return

    # Group by model
    models = sorted(set(r['model'] for r in results))

    for model in models:
        model_results = [r for r in results if r['model'] == model]

        # Group by variant
        variants = {}
        for r in model_results:
            v = r['variant']
            if v not in variants:
                variants[v] = {}
            variants[v][r['metric']] = r.get('score', 0)
            if 'correct' in r:
                variants[v]['correct'] = r['correct']
                variants[v]['total'] = r['total']

        print(f"\n  {model}:")
        if task_name == 'TruthfulQA':
            print(f"    {'Variant':<20} {'MC1':<10} {'MC2':<10}")
            print(f"    {'-'*40}")
            for v in ['v', 'rope', 'v_rope']:
                if v in variants:
                    mc1 = variants[v].get('MC1', 0)
                    mc2 = variants[v].get('MC2', 0)
                    print(f"    {v:<20} {mc1:<10.4f} {mc2:<10.4f}")
        else:
            print(f"    {'Variant':<20} {'EM Accuracy':<15} {'Correct/Total'}")
            print(f"    {'-'*50}")
            for v in ['v', 'rope', 'v_rope']:
                if v in variants:
                    em = variants[v].get('EM', 0)
                    correct = variants[v].get('correct', '?')
                    total = variants[v].get('total', '?')
                    print(f"    {v:<20} {em:<15.4f} {correct}/{total}")


if __name__ == '__main__':
    print("=" * 60)
    print("LoFiT-RoPE Results Summary")
    print("=" * 60)

    all_results = {}

    # TruthfulQA
    tqa = compute_tqa_scores()
    if tqa:
        all_results['TruthfulQA'] = tqa
        print("\n--- TruthfulQA (MC1 / MC2) ---")
        print_task_results('TruthfulQA', tqa)
    else:
        print("\n--- TruthfulQA: No results found ---")

    # MQuAKE
    mquake = compute_json_scores('mquake')
    if mquake:
        all_results['MQuAKE'] = mquake
        print("\n--- MQuAKE (Exact Match) ---")
        print_task_results('MQuAKE', mquake)
    else:
        print("\n--- MQuAKE: No results found ---")

    # CLUTRR
    clutrr = compute_json_scores('clutrr')
    if clutrr:
        all_results['CLUTRR'] = clutrr
        print("\n--- CLUTRR (Exact Match) ---")
        print_task_results('CLUTRR', clutrr)
    else:
        print("\n--- CLUTRR: No results found ---")

    # Final comparison table
    print("\n" + "=" * 60)
    print("Cross-Task Comparison")
    print("=" * 60)
    print(f"\n{'Task':<15} {'Model':<15} {'Variant':<12} {'Metric':<8} {'Score':<10}")
    print("-" * 60)

    for task_name, results in all_results.items():
        for r in sorted(results, key=lambda x: (x['model'], x['variant'], x['metric'])):
            if r['variant'] in ['v', 'rope', 'v_rope']:
                print(f"{r['task']:<15} {r['model']:<15} {r['variant']:<12} {r['metric']:<8} {r['score']:<10.4f}")

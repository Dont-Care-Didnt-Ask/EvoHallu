#!/usr/bin/env python3
"""
Build LaTeX table from PRR evaluation results.
Generates a table with methods as rows, datasets as columns, and average rank.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
import argparse


def build_dataset_metrics_table(model_dirs, output_file=None):
    """
    Build a table with datasets+metrics as rows and models as columns.

    Args:
        model_dirs: Dictionary mapping model names to their metric data directories
                   e.g., {'Llama 3.1 8B Instruct': 'path/to/llama/metrics',
                          'Qwen3-8B': 'path/to/qwen/metrics'}
        output_file: Optional path to save the LaTeX table

    Returns:
        DataFrame with the results and latex_table string
    """
    # Define dataset grouping and order
    qa_datasets = ['babi_qa_continuation', 'coqa_simple_instruct', 'mmlu_simple_instruct',
                   'triviaqa_simple_instruct', 'truthfulqa_simple_instruct']
    gsm8k_datasets = ['gsm8k_simple_instruct']
    translation_datasets = ['wmt14_deen', 'wmt14_fren', 'wmt19_deen_simple_instruct',
                           'wmt19_ruen_simple_instruct']
    summarization_datasets = ['samsum_simple_instruct', 'xsum_simple_instruct']

    all_datasets = qa_datasets + gsm8k_datasets + translation_datasets + summarization_datasets

    # Define dataset type mapping
    dataset_types = {}
    for ds in qa_datasets:
        dataset_types[ds] = 'QA'
    for ds in gsm8k_datasets:
        dataset_types[ds] = 'Math'
    for ds in translation_datasets:
        dataset_types[ds] = 'Translation'
    for ds in summarization_datasets:
        dataset_types[ds] = 'Summarization'

    # Dataset name mapping
    dataset_display_names = {
        'babi_qa_continuation': 'BaBI',
        'coqa_simple_instruct': 'CoQA',
        'mmlu_simple_instruct': 'MMLU',
        'triviaqa_simple_instruct': 'Trivia',
        'truthfulqa_simple_instruct': 'Truthful',
        'gsm8k_simple_instruct': 'GSM8K',
        'wmt14_deen': 'WMT14$_{de}$',
        'wmt14_fren': 'WMT14$_{fr}$',
        'wmt19_deen_simple_instruct': 'WMT19$_{de}$',
        'wmt19_ruen_simple_instruct': 'WMT19$_{ru}$',
        'samsum_simple_instruct': 'SAMSum',
        'xsum_simple_instruct': 'XSum'
    }

    # Define which metrics to use for each dataset type
    # These are the exact metrics used for PRR computation
    dataset_metrics = {
        # QA datasets
        'babi_qa_continuation': 'exact_match',
        'coqa_simple_instruct': 'f1',
        'mmlu_simple_instruct': 'exact_match',
        'triviaqa_simple_instruct': 'f1',
        'truthfulqa_simple_instruct': 'best_rougeL',
        # GSM8K
        'gsm8k_simple_instruct': 'exact_match',
        # Translation datasets use BLEU
        'wmt14_deen': 'bleu',
        'wmt14_fren': 'bleu',
        'wmt19_deen_simple_instruct': 'bleu',
        'wmt19_ruen_simple_instruct': 'bleu',
        # Summarization datasets use ROUGE-L
        'samsum_simple_instruct': 'rougeL',
        'xsum_simple_instruct': 'rougeL'
    }

    # Metric name mapping for display (full names)
    metric_display_names = {
        'exact_match': 'Exact Match',
        'f1': 'F1',
        'best_rougeL': 'ROUGE-L',
        'rouge1': 'ROUGE-1',
        'rouge2': 'ROUGE-2',
        'rougeL': 'ROUGE-L',
        'bleu': 'BLEU'
    }

    # Collect data for each model
    results = {}

    for model_name, model_dir in model_dirs.items():
        model_path = Path(model_dir)
        model_data = {}

        # Read CSV files for each dataset
        for dataset in all_datasets:
            csv_file = model_path / f"{dataset}.csv"

            # Skip PARTIAL files
            if 'PARTIAL_' in dataset:
                continue

            if not csv_file.exists():
                continue

            try:
                df_data = pd.read_csv(csv_file)
                metric_col = dataset_metrics.get(dataset)

                if metric_col and metric_col in df_data.columns:
                    # Calculate mean and convert to percentage
                    metric_value = df_data[metric_col].mean() * 100
                    model_data[dataset] = {
                        'metric': metric_col,
                        'value': metric_value
                    }
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")

        results[model_name] = model_data

    # Build the table
    rows = []

    for dataset in all_datasets:
        # Check if any model has data for this dataset
        has_data = any(dataset in model_data for model_data in results.values())
        if not has_data:
            continue

        metric = dataset_metrics.get(dataset, 'unknown')
        dataset_display = dataset_display_names.get(dataset, dataset)
        metric_display = metric_display_names.get(metric, metric)
        dataset_type = dataset_types.get(dataset, 'Unknown')

        # Create row with dataset name, type, metric, and model values
        row = {'Dataset': dataset_display, 'Type': dataset_type, 'Metric': metric_display}

        for model_name in model_dirs.keys():
            if dataset in results[model_name]:
                row[model_name] = results[model_name][dataset]['value']
            else:
                row[model_name] = None

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Set Dataset as index for cleaner display (optional)
    # df = df.set_index('Dataset')

    # Generate LaTeX table manually to place caption at bottom
    latex_table = df.to_latex(
        float_format="%.1f",
        na_rep='-',
        index=False,  # Don't include the default index
        escape=False  # Don't escape $ signs in column names
    )

    # Move caption to bottom by reconstructing the table
    # Remove the default table environment and add our own with caption at bottom
    lines = latex_table.split('\n')
    # Find tabular content (skip empty lines)
    tabular_start = next(i for i, line in enumerate(lines) if '\\begin{tabular}' in line)
    tabular_end = next(i for i, line in enumerate(lines) if '\\end{tabular}' in line)

    tabular_content = '\n'.join(lines[tabular_start:tabular_end+1])

    latex_table = f"""\\begin{{table}}
{tabular_content}
\\caption{{Dataset metrics across different models}}
\\label{{tab:dataset_metrics}}
\\end{{table}}
"""

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"Dataset metrics table saved to: {output_file}")

    print(latex_table)
    return df, latex_table


def build_aggregated_prr_table(directory, output_file=None):
    """
    Build a table with methods as rows and dataset groups as columns.
    Groups: QA, Summarization, Machine Translation, plus overall Average.
    """
    # Directory containing JSON files
    json_dir = Path(f'out/logs/example/EVALS_polygraph/{directory}')

    # Read all JSON files
    data = {}
    for json_file in sorted(json_dir.glob('*.json')):
        dataset_name = json_file.stem  # filename without extension
        with open(json_file, 'r') as f:
            content = json.load(f)
            data[dataset_name] = content

    # Extract PRR values for each estimator and dataset
    methods = set()
    for dataset_name, content in data.items():
        if 'estimators' in content:
            methods.update(content['estimators'].keys())

    # Create a dictionary to store the results
    results = {method: {} for method in methods}

    for dataset_name, content in data.items():
        if 'estimators' in content:
            for method, metrics in content['estimators'].items():
                prr_value = metrics.get('PRR')
                results[method][dataset_name] = prr_value

    # Define dataset grouping
    qa_datasets = {'babi_qa_continuation', 'coqa_simple_instruct', 'mmlu_simple_instruct',
                   'triviaqa_simple_instruct', 'truthfulqa_simple_instruct', 'gsm8k_simple_instruct'}
    summarization_datasets = {'samsum_simple_instruct', 'xsum_simple_instruct'}
    mt_datasets = {'wmt14_deen', 'wmt14_fren', 'wmt19_deen_simple_instruct',
                   'wmt19_ruen_simple_instruct'}
    n_existing_qa_datasets = len([k for k in qa_datasets if k in data])
    n_existing_summarization_datasets = len([k for k in summarization_datasets if k in data])
    n_existing_mt_datasets = len([k for k in mt_datasets if k in data])

    columns = [
        'Average', 
        rf'\makecell{{QA \\ ({n_existing_qa_datasets} datasets)}}',
        rf'\makecell{{Summarization \\ ({n_existing_summarization_datasets} datasets)}}',
        rf'\makecell{{Machine Translation \\ ({n_existing_mt_datasets} datasets)}}'
    ]

    # Build aggregated results
    aggregated_results = {}
    for method in methods:
        method_data = results[method]
        
        # Calculate average for each group
        qa_values = [v for k, v in method_data.items() if k in qa_datasets and v is not None]
        sum_values = [v for k, v in method_data.items() if k in summarization_datasets and v is not None]
        mt_values = [v for k, v in method_data.items() if k in mt_datasets and v is not None]
        
        qa_avg = np.mean(qa_values) if qa_values else None
        sum_avg = np.mean(sum_values) if sum_values else None
        mt_avg = np.mean(mt_values) if mt_values else None
        
        # Overall average (mean of 3 group averages)
        group_avgs = [v for v in [qa_avg, sum_avg, mt_avg] if v is not None]
        overall_avg = np.mean(group_avgs) if group_avgs else None
        
        aggregated_results[method] = dict(zip(columns, [overall_avg, qa_avg, sum_avg, mt_avg]))

    # Create DataFrame
    df = pd.DataFrame(aggregated_results).T
    
    # Define column order
    df = df[columns]

    # Sort by average (descending, higher PRR is better)
    df = df.sort_values('Average', ascending=False)

    # Shorten method names to max 25 characters and ensure uniqueness
    seen_names = {}
    new_index = []
    for name in df.index:
        short_name = name[:25]
        if short_name in seen_names:
            seen_names[short_name] += 1
            short_name = f"{short_name[:23]}{seen_names[short_name]}"
        else:
            seen_names[short_name] = 0
        new_index.append(short_name)
    df.index = new_index

    # Define existing methods (before escaping underscores)
    existing_methods = {
        'MeanStdTokenEntropy',
        'MeanTokenEntropy',
        'MaximumSequenceProbabilit',  # Truncated version
        'Perplexity',
        'StdTokenEntropy',
        'SequenceLogProb',
        'MinimalLogProb',
    }
    out_of_competition_methods = {
        'RAUQ',
        'AttentionScore (layer=16)'
    }

    # Color method names and escape underscores for LaTeX rendering
    colored_index = []
    for name in df.index:
        escaped_name = name.replace('_', r'\_')
        if name in existing_methods:
            # Orange for existing methods
            colored_name = f"\\textcolor{{orange}}{{{escaped_name}}}"
        elif name in out_of_competition_methods:
            # Gray for out-of-competition methods
            colored_name = f"\\textcolor{{gray}}{{{escaped_name}}}"
        else:
            # Blue for evolved methods
            colored_name = f"\\textcolor{{blue}}{{{escaped_name}}}"
        colored_index.append(colored_name)
    df.index = colored_index

    # Format top-1 (bold) and top-2 (underlined) for each column
    # FIX: Use unique values to handle ties correctly
    df_formatted = df.astype(object)

    for col in df.columns:
        # Get non-null values and their indices
        col_values = df[col].dropna()
        if len(col_values) == 0:
            continue

        # Find top-1 and top-2 values using UNIQUE values (fixes tie issue)
        rounded_values = col_values.round(1)
        sorted_rounded = rounded_values.sort_values(ascending=False)
        unique_values = sorted_rounded.unique()

        top1_indices = []
        top2_indices = []

        if len(unique_values) >= 1:
            top1_value = unique_values[0]
            top1_indices = rounded_values[rounded_values.apply(lambda x: abs(x - top1_value) < 0.01)].index.tolist()

        if len(unique_values) >= 2:
            top2_value = unique_values[1]  # Second UNIQUE value
            top2_indices = rounded_values[rounded_values.apply(lambda x: abs(x - top2_value) < 0.01)].index.tolist()

        # Format the column
        for idx in df.index:
            val = df.loc[idx, col]
            if pd.notna(val):
                if idx in top1_indices:
                    df_formatted.loc[idx, col] = f"\\textbf{{{val:.1f}}}"
                elif idx in top2_indices:
                    df_formatted.loc[idx, col] = f"\\underline{{{val:.1f}}}"
                else:
                    df_formatted.loc[idx, col] = f"{val:.1f}"
            else:
                df_formatted.loc[idx, col] = '-'

    # Generate LaTeX table
    latex_table = df_formatted.to_latex(
        na_rep='-',
        escape=False,  # Don't escape LaTeX commands
    )

    # Move caption to bottom by reconstructing the table
    lines = latex_table.split('\n')
    tabular_start = next(i for i, line in enumerate(lines) if '\\begin{tabular}' in line)
    tabular_end = next(i for i, line in enumerate(lines) if '\\end{tabular}' in line)

    tabular_content = '\n'.join(lines[tabular_start:tabular_end+1])

    latex_table = f"""\\begin{{table}}
{tabular_content}
\\caption{{PRR scores across different methods and dataset groups. \\textcolor{{blue}}{{Blue}}: evolved methods, \\textcolor{{orange}}{{orange}}: existing methods, \\textcolor{{gray}}{{gray}}: out-of-competition. \\textbf{{Bold}}: best, \\underline{{underlined}}: second best.}}
\\label{{tab:prr_aggregated}}
\\end{{table}}
"""

    print(latex_table)

    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"\nTable saved to: {output_file}")
    else:
        output_file = Path('prr_aggregated_table.tex')
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"\nTable saved to: {output_file}")


def build_prr_table(directory):
    # Directory containing JSON files
    json_dir = Path(f'out/logs/example/EVALS_polygraph/{directory}')

    # Read all JSON files
    data = {}
    for json_file in sorted(json_dir.glob('*.json')):
        dataset_name = json_file.stem  # filename without extension
        with open(json_file, 'r') as f:
            content = json.load(f)
            data[dataset_name] = content

    # Extract PRR values for each estimator and dataset
    methods = set()
    for dataset_name, content in data.items():
        if 'estimators' in content:
            methods.update(content['estimators'].keys())

    # Create a dictionary to store the results
    results = {method: {} for method in methods}

    for dataset_name, content in data.items():
        if 'estimators' in content:
            for method, metrics in content['estimators'].items():
                prr_value = metrics.get('PRR')
                results[method][dataset_name] = prr_value

    # Create DataFrame
    df = pd.DataFrame(results).T

    # Define dataset grouping and order
    qa_datasets = ['babi_qa_continuation', 'coqa_simple_instruct', 'mmlu_simple_instruct',
                   'triviaqa_simple_instruct', 'truthfulqa_simple_instruct']
    gsm8k_datasets = ['gsm8k_simple_instruct']
    translation_datasets = ['wmt14_deen', 'wmt14_fren', 'wmt19_deen_simple_instruct',
                           'wmt19_ruen_simple_instruct']
    summarization_datasets = ['samsum_simple_instruct', 'xsum_simple_instruct']

    # Order columns according to grouping
    ordered_columns = []
    for dataset in qa_datasets + gsm8k_datasets + translation_datasets + summarization_datasets:
        if dataset in df.columns:
            ordered_columns.append(dataset)

    # Reorder columns
    df = df[ordered_columns]

    # Shorten column names for better display with makecell for multi-line headers
    column_rename = {
        'babi_qa_continuation': 'BaBI',
        'coqa_simple_instruct': 'CoQA',
        'mmlu_simple_instruct': 'MMLU',
        'triviaqa_simple_instruct': 'Trivia',
        'truthfulqa_simple_instruct': r'\makecell{Truth\\fulQA}',
        'gsm8k_simple_instruct': 'GSM8K',
        'wmt14_deen': r'\makecell{WMT\\14 de}',
        'wmt14_fren': r'\makecell{WMT\\14 fr}',
        'wmt19_deen_simple_instruct': r'\makecell{WMT\\19 de}',
        'wmt19_ruen_simple_instruct': r'\makecell{WMT\\19 ru}',
        'samsum_simple_instruct': 'SAMSum',
        'xsum_simple_instruct': 'XSum'
    }

    df = df.rename(columns=column_rename)

    average_prr = df.mean(axis=1)
    average_prr_col = r'\makecell{Avg\\PRR}'
    # Calculate ranks for each dataset (higher PRR is better, so rank 1 = highest PRR)
    # Use ascending=False because higher PRR is better
    ranks = df.rank(ascending=False, na_option='keep')

    # Calculate average rank across all datasets
    avg_rank = ranks.mean(axis=1)

    # Add average rank column to the original dataframe with makecell for multi-line header
    df[r'\makecell{Avg\\Rank}'] = avg_rank
    df[average_prr_col] = average_prr

    # Sort by average rank (ascending, so best methods first)
    avg_rank_col = r'\makecell{Avg\\Rank}'
    df = df.sort_values(avg_rank_col)

    # Reorder columns to put Avg Rank first (second column in table after method names)
    cols = [avg_rank_col] + [average_prr_col] + [col for col in df.columns if col != avg_rank_col and col != average_prr_col]
    df = df[cols]

    # Shorten method names to max 25 characters and ensure uniqueness
    seen_names = {}
    new_index = []
    for name in df.index:
        short_name = name[:25]
        if short_name in seen_names:
            seen_names[short_name] += 1
            short_name = f"{short_name[:23]}{seen_names[short_name]}"
        else:
            seen_names[short_name] = 0
        new_index.append(short_name)
    df.index = new_index

    # Define existing methods (before escaping underscores)
    existing_methods = {
        'MeanStdTokenEntropy',
        'MeanTokenEntropy',
        'MaximumSequenceProbabilit',  # Truncated version
        'Perplexity',
        'StdTokenEntropy',
        'SequenceLogProb',
        'MinimalLogProb',
    }
    out_of_competition_methods = {
        'RAUQ',
        'AttentionScore (layer=16)'
    }

    # Color method names and escape underscores for LaTeX rendering
    colored_index = []
    for name in df.index:
        escaped_name = name.replace('_', r'\_')
        if name in existing_methods:
            # Orange for existing methods
            colored_name = f"\\textcolor{{orange}}{{{escaped_name}}}"
        elif name in out_of_competition_methods:
            # Gray for out-of-competition methods
            colored_name = f"\\textcolor{{gray}}{{{escaped_name}}}"
        else:
            # Blue for evolved methods
            colored_name = f"\\textcolor{{blue}}{{{escaped_name}}}"
        colored_index.append(colored_name)
    df.index = colored_index

    # Format top-1 (bold) and top-2 (underlined) for each column (excluding Avg Rank)
    # Create a copy as object dtype to allow string formatting
    df_formatted = df.astype(object)

    for col in df.columns:
        if col == avg_rank_col:
            continue

        # Get non-null values and their indices
        col_values = df[col].dropna()
        if len(col_values) == 0:
            continue

        # ROUND values to 1 decimal before finding top indices (matches display format)
        rounded_values = col_values.round(1)
        sorted_rounded = rounded_values.sort_values(ascending=False)
        unique_values = sorted_rounded.unique()

        top1_indices = []
        top2_indices = []

        if len(unique_values) >= 1:
            top1_value = unique_values[0]
            top1_indices = rounded_values[rounded_values.apply(lambda x: abs(x - top1_value) < 0.01)].index.tolist()

        if len(unique_values) >= 2:
            top2_value = unique_values[1]  # Second UNIQUE rounded value
            top2_indices = rounded_values[rounded_values.apply(lambda x: abs(x - top2_value) < 0.01)].index.tolist()

        # Format the column
        for idx in df.index:
            val = df.loc[idx, col]
            if pd.notna(val):
                if idx in top1_indices:
                    df_formatted.loc[idx, col] = f"\\textbf{{{val:.1f}}}"
                elif idx in top2_indices:
                    df_formatted.loc[idx, col] = f"\\underline{{{val:.1f}}}"
                else:
                    df_formatted.loc[idx, col] = f"{val:.1f}"
            else:
                df_formatted.loc[idx, col] = '-'

    # Format Avg Rank column (no bold/underline)
    for idx in df.index:
        val = df.loc[idx, avg_rank_col]
        df_formatted.loc[idx, avg_rank_col] = f"{val:.1f}" if pd.notna(val) else '-'

    # Generate LaTeX table
    latex_table = df_formatted.to_latex(
        na_rep='-',
        escape=False,  # Don't escape LaTeX commands
    )

    # Move caption to bottom by reconstructing the table
    lines = latex_table.split('\n')
    tabular_start = next(i for i, line in enumerate(lines) if '\\begin{tabular}' in line)
    tabular_end = next(i for i, line in enumerate(lines) if '\\end{tabular}' in line)

    tabular_content = '\n'.join(lines[tabular_start:tabular_end+1])

    latex_table = f"""\\begin{{table}}
{tabular_content}
\\caption{{PRR scores across different methods and datasets (sorted by average rank). \\textcolor{{blue}}{{Blue}}: evolved methods, \\textcolor{{orange}}{{orange}}: existing methods.}}
\\label{{tab:prr_scores}}
\\end{{table}}
"""

    print(latex_table)

    # Save to file
    output_file = Path('prr_results_table.tex')
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\nTable saved to: {output_file}")


if __name__ == '__main__':
    import sys

    directory = "Feb25_all_prr_050_length_filtered"

    parser = argparse.ArgumentParser(description='Build PRR evaluation tables')
    parser.add_argument('--metrics', action='store_true', 
                        help='Build dataset metrics table')
    parser.add_argument('--aggregated', action='store_true',
                        help='Build aggregated PRR table with dataset groups')
    parser.add_argument('--directory', type=str, default=directory,
                        help='Directory containing evaluation results')
    
    args = parser.parse_args()

    # Default: build PRR table
    if not args.metrics and not args.aggregated:
        build_prr_table(args.directory)
    # Build dataset metrics table
    elif args.metrics:
        model_dirs = {
            'Llama 3.1 8B Instruct': '/mnt/extremessd10tb/seleznev/EvoTune/data/uncertainty_quantification/polygraph_generations_metrics',
            'Qwen3-8B': '/mnt/extremessd10tb/seleznev/EvoTune/data/uncertainty_quantification/polygraph_generations_qwen3_8b_metrics'
        }
        output_file = Path('/mnt/extremessd10tb/seleznev/EvoTune/dataset_metrics_table.tex')
        build_dataset_metrics_table(model_dirs, output_file)
    # Build aggregated PRR table
    elif args.aggregated:
        output_file = Path('prr_aggregated_table.tex')
        build_aggregated_prr_table(args.directory, output_file)
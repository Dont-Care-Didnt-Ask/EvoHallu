#!/usr/bin/env python3
"""
Fact Checker using VLLM
Classifies claims as True or False based on factual accuracy.

Usage:
    python fact_checker.py input.csv output.csv --claim-column "claim" --model "meta-llama/Llama-3.1-8B-Instruct" --num-examples 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import numpy as np
from sklearn.metrics import roc_auc_score


def setup_environment(cache_dir="/mnt/extremessd10tb/seleznev/.cache"):
    """Set up environment variables for HuggingFace."""
    os.environ['HF_HOME'] = cache_dir
    # Note: Replace with your own HF token or set via environment variable
    if 'HF_TOKEN' not in os.environ:
        print("Warning: HF_TOKEN not set. You may need to set it for private models.")


def initialize_model(model_name, tensor_parallel_size=2, gpu_memory_util=0.94, 
                    max_model_len=2500, max_num_seqs=64, cache_dir="/mnt/extremessd10tb/seleznev/.cache"):
    """Initialize VLLM model with specified parameters."""
    print(f"Initializing model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_util,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        download_dir=cache_dir,
    )
    return llm


def get_few_shot_examples():
    """Return all available few-shot examples for fact-checking."""
    examples = [
        # Example 1: False historical claim
        {
            "user": "Claim: Christopher Columbus discovered America in 1592.",
            "assistant": "Columbus made his first voyage to the Americas in 1492, not 1592 - this is off by 100 years; FALSE"
        },
        # Example 2: True scientific fact
        {
            "user": "Claim: Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.",
            "assistant": "This is the correct boiling point of water at standard atmospheric pressure (1 atm) at sea level; TRUE"
        },
        # Example 3: False geographical claim
        {
            "user": "Claim: Brazil is the largest country in South America and shares borders with every South American country.",
            "assistant": "Brazil is the largest country in South America but does not border Chile and Ecuador, so it doesn't share borders with every South American country; FALSE"
        },
        # Example 4: True mathematical fact
        {
            "user": "Claim: The square root of 144 is 12.",
            "assistant": "12 × 12 equals 144, so the square root of 144 is indeed 12; TRUE"
        },
        # Example 5: False biological claim
        {
            "user": "Claim: Humans have 23 chromosomes in total.",
            "assistant": "Humans have 23 pairs of chromosomes for a total of 46 chromosomes, not 23; FALSE"
        },
        # Example 6: True historical fact
        {
            "user": "Claim: World War II ended in 1945 with the surrender of Japan.",
            "assistant": "Japan formally surrendered on September 2, 1945, ending World War II; TRUE"
        },
        # Example 7: False astronomical claim
        {
            "user": "Claim: The Sun revolves around the Earth once every 24 hours.",
            "assistant": "The Earth rotates on its axis every 24 hours and revolves around the Sun, not the other way around - this describes the geocentric model which was disproven; FALSE"
        },
        # Example 8: True chemical fact
        {
            "user": "Claim: The chemical symbol for gold is Au, derived from its Latin name 'aurum'.",
            "assistant": "Au is indeed the chemical symbol for gold, derived from the Latin word 'aurum' meaning gold; TRUE"
        },
        # Example 9: False political claim
        {
            "user": "Claim: The United Nations was founded in 1950 after World War II.",
            "assistant": "The United Nations was founded in 1945, not 1950, though it was indeed after World War II; FALSE"
        },
        # Example 10: True geographical fact
        {
            "user": "Claim: Mount Everest, located on the border between Nepal and Tibet, is the highest mountain above sea level on Earth.",
            "assistant": "Mount Everest at 8,848.86 meters (29,031.7 feet) is indeed the highest mountain above sea level and sits on the Nepal-Tibet border; TRUE"
        }
    ]
    return examples


def get_classification_examples():
    """Return additional domain-specific examples for different types of claims."""
    domain_examples = {
        "medical": [
            {
                "user": "Claim: Type 1 diabetes is caused by the body's immune system attacking insulin-producing cells in the pancreas.",
                "assistant": "Type 1 diabetes is indeed an autoimmune condition where the immune system destroys beta cells in the pancreas that produce insulin; TRUE"
            },
            {
                "user": "Claim: Antibiotics are effective against viral infections like the common cold.",
                "assistant": "Antibiotics only work against bacterial infections, not viral infections like the common cold or flu; FALSE"
            }
        ],
        "technology": [
            {
                "user": "Claim: The first iPhone was released by Apple in 2007.",
                "assistant": "Steve Jobs announced and Apple released the first iPhone on June 29, 2007; TRUE"
            },
            {
                "user": "Claim: Quantum computers use traditional binary bits that can be either 0 or 1.",
                "assistant": "Quantum computers use quantum bits (qubits) that can exist in superposition of both 0 and 1 states simultaneously, not traditional binary bits; FALSE"
            }
        ],
        "sports": [
            {
                "user": "Claim: A marathon race is exactly 26.2 miles or 42.195 kilometers long.",
                "assistant": "The official marathon distance is 26.219 miles or 42.195 kilometers, commonly rounded to 26.2 miles; TRUE"
            },
            {
                "user": "Claim: In soccer, a team consists of 10 players including the goalkeeper.",
                "assistant": "A soccer team consists of 11 players on the field including the goalkeeper, not 10; FALSE"
            }
        ]
    }
    return domain_examples


def create_prompt(claim, tokenizer, model_name, num_examples=4, domain=None, include_confidence=False):
    """Create a prompt for fact-checking with configurable number of examples."""
    
    # System message
    system_content = """You are an expert fact-checker with comprehensive knowledge across all domains. Your task is to determine whether claims are factually TRUE or FALSE based on established facts and evidence.\
Rules:\
- Evaluate the factual accuracy of each claim based on verifiable information\
- Consider the claim as written - small errors in dates, numbers, or facts make a claim FALSE\
- If a claim contains multiple parts, all parts must be true for the claim to be TRUE\
- If any part of the claim is incorrect, misleading, or inaccurate, classify it as FALSE\
- Provide a brief explanation (1-2 sentences) of your reasoning\
- Output exactly in this format: explanation followed by a semicolon and either "TRUE" or "FALSE"\
"""

    if include_confidence:
        system_content += "- You may optionally add a confidence level (HIGH/MEDIUM/LOW) after your classification"

    messages = [
        {
            "role": "system",
            "content": system_content
        }
    ]
    
    # Add few-shot examples based on num_examples parameter
    if num_examples > 0:
        all_examples = get_few_shot_examples()
        
        # Add domain-specific examples if requested
        if domain and domain in get_classification_examples():
            domain_examples = get_classification_examples()[domain]
            all_examples.extend(domain_examples)
        
        selected_examples = all_examples[:min(num_examples, len(all_examples))]
        
        for example in selected_examples:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})
    
    # Add the actual claim to validate
    messages.append({"role": "user", "content": f"Claim: {claim}"})
    
    # Check if model supports thinking mode
    enable_thinking = "Llama-3" in model_name or "llama-3" in model_name
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt


def process_outputs(outputs, include_confidence=False):
    """Process model outputs to extract classifications."""
    processed = []
    clean_labels = []
    is_true = []
    confidence_levels = []
    
    for output in outputs:
        text = output.outputs[0].text.strip()
        
        # Remove thinking tags if present
        if 'think>' in text:
            text = text.split('think>')[-1].strip()
        
        processed.append(text)
        
        # Extract label (TRUE/FALSE)
        if ";" in text:
            parts = text.split(";")
            label = parts[-1].strip()
        else:
            label = text
        
        clean_labels.append(label)
        
        # Boolean classification
        is_true.append('TRUE' in label.upper())
        
        # Extract confidence if present
        if include_confidence:
            confidence = "UNKNOWN"
            for level in ["HIGH", "MEDIUM", "LOW"]:
                if level in label.upper():
                    confidence = level
                    break
            confidence_levels.append(confidence)
    
    if include_confidence:
        return processed, clean_labels, is_true, confidence_levels
    else:
        return processed, clean_labels, is_true

def load_prompt_from_file(file_path):
    """Load prompt from a file."""
    with open(file_path, 'r') as file:
        json_data = json.load(file)

        

def fact_check(input_file, output_file, claim_column='claim', 
               model_name="meta-llama/Llama-3.1-8B-Instruct",
               batch_size=None, temperature=0.3, max_tokens=150,
               num_examples=4, domain=None, include_confidence=False,
               tensor_parallel_size=2, gpu_memory_util=0.94, filtered=False,
               max_model_len=2500, cache_dir="/.cache/"):
    """Main function to fact-check claims in a CSV file."""
    
    # Setup environment
    setup_environment(cache_dir)
    
    # Load data
    print(f"Loading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
        if filtered:
            print(f"with Filter")
            df = df[df['is_good'] == True].copy()

    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Validate column exists
    if claim_column not in df.columns:
        print(f"Error: Column '{claim_column}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Found {len(df)} rows to process")
    print(f"Using {num_examples} few-shot examples")
    if domain:
        print(f"Using domain-specific examples for: {domain}")
    if include_confidence:
        print("Including confidence scores in output")
    
    # Adjust max_model_len based on number of examples
    if num_examples > 6:
        suggested_len = max(max_model_len, 3500)
        if suggested_len != max_model_len:
            print(f"Note: Adjusting max_model_len from {max_model_len} to {suggested_len} for {num_examples} examples")
            max_model_len = suggested_len
    
    # Initialize model and tokenizer
    llm = initialize_model(model_name, tensor_parallel_size, gpu_memory_util, 
                          max_model_len, 64, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Create prompts
    print("Creating prompts...")
    prompts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        claim = row[claim_column]
        if pd.notna(claim):  # Skip NaN values
            prompt = create_prompt(claim, tokenizer, model_name, num_examples, 
                                 domain, include_confidence)
            prompts.append(prompt)
        else:
            prompts.append("")  # Placeholder for NaN
    
    df['prompt'] = prompts
    
    # Filter out empty prompts for generation
    valid_indices = [i for i, p in enumerate(prompts) if p]
    valid_prompts = [prompts[i] for i in valid_indices]
    
    # Generate responses
    print(f"Generating fact-checks for {len(valid_prompts)} valid claims...")
    if batch_size:
        # Process in batches if specified
        all_outputs = []
        for i in tqdm(range(0, len(valid_prompts), batch_size)):
            batch = valid_prompts[i:i+batch_size]
            batch_outputs = llm.generate(batch, sampling_params)
            all_outputs.extend(batch_outputs)
        outputs = all_outputs
    else:
        outputs = llm.generate(valid_prompts, sampling_params)
    
    # Process outputs
    print("Processing outputs...")
    if include_confidence:
        processed, clean_labels, is_true, confidence_levels = process_outputs(outputs, include_confidence=True)
    else:
        processed, clean_labels, is_true = process_outputs(outputs, include_confidence=False)
    
    # Map results back to dataframe (accounting for NaN values)
    df['explanation'] = ""
    df['verdict'] = ""
    df['is_true'] = None
    if include_confidence:
        df['confidence'] = ""
    
    for idx, orig_idx in enumerate(valid_indices):
        df.loc[orig_idx, 'explanation'] = processed[idx]
        df.loc[orig_idx, 'verdict'] = clean_labels[idx]
        df.loc[orig_idx, 'is_true'] = is_true[idx]
        if include_confidence:
            df.loc[orig_idx, 'confidence'] = confidence_levels[idx]
    
    # Remove prompt column before saving (it's usually too large)
    df = df.drop(columns=['prompt'])
    
    # Save results
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    total_processed = len(valid_indices)
    total_true = sum(is_true)
    total_false = total_processed - total_true
    
    print("\n" + "="*50)
    print("FACT-CHECKING SUMMARY")
    print("="*50)
    print(f"Total claims: {len(df)}")
    print(f"Processed: {total_processed}")
    print(f"Skipped (NaN): {len(df) - total_processed}")
    print(f"TRUE: {total_true} ({total_true/total_processed*100:.1f}%)")
    print(f"FALSE: {total_false} ({total_false/total_processed*100:.1f}%)")
    
    if include_confidence and confidence_levels:
        high_conf = confidence_levels.count("HIGH")
        med_conf = confidence_levels.count("MEDIUM")
        low_conf = confidence_levels.count("LOW")
        print(f"\nConfidence Distribution:")
        print(f"  HIGH: {high_conf} ({high_conf/total_processed*100:.1f}%)")
        print(f"  MEDIUM: {med_conf} ({med_conf/total_processed*100:.1f}%)")
        print(f"  LOW: {low_conf} ({low_conf/total_processed*100:.1f}%)")
    
    print(f"\nOutput saved to: {output_file}")
    
    # Print sample results
    print("\nSample Results (first 3):")
    print("-" * 50)
    sample_df = df[df['is_true'].notna()].head(3)
    for idx, row in sample_df.iterrows():
        print(f"Claim: {row[claim_column][:80]}...")
        print(f"Verdict: {row['verdict']}")
        if include_confidence and 'confidence' in row:
            print(f"Confidence: {row['confidence']}")
        print("-" * 50)

    if 'is_hallu' in df.columns:
        valid_df = df[df['is_true'].notna()]
        if len(valid_df) == 0:
            print("No valid predictions for ROC-AUC calculation")
        else:
            true_labels = valid_df['is_hallu'].values
            if include_confidence:
                conf_to_num = {'HIGH': 0.9, 'MEDIUM': 0.5, 'LOW': 0.1}
                print("Computing ROC-AUC with manual mapping of confidence scores:", conf_to_num)
                conf_num = valid_df['confidence'].map(conf_to_num).fillna(0.5)
                pred_scores = np.where(valid_df['is_true'], 1 - conf_num, conf_num)
            else:
                print("Computing ROC-AUC from 0/1 predictions")
                pred_scores = (~valid_df['is_true']).astype(float)
            if len(set(true_labels)) < 2:
                print("Cannot compute ROC-AUC: all true labels are the same")
            else:
                auc = roc_auc_score(true_labels, pred_scores)
                print(f"ROC-AUC (hallu as positive): {auc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fact-check claims as TRUE or FALSE using VLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default 4 examples
    python fact_checker.py input.csv output.csv --claim-column "claim"
    
    # Zero-shot (no examples)
    python fact_checker.py input.csv output.csv --claim-column "claim" --num-examples 0
    
    # With maximum 10 examples
    python fact_checker.py input.csv output.csv --claim-column "statement" --num-examples 10
    
    # With confidence scores
    python fact_checker.py input.csv output.csv --claim-column "fact" --include-confidence
    
    # With domain-specific examples
    python fact_checker.py input.csv output.csv --claim-column "claim" --domain medical
    
    # With batch processing
    python fact_checker.py input.csv output.csv --claim-column "claim" --batch-size 32 --num-examples 6
    
    # List available examples
    python fact_checker.py --list-examples
    
    # List domain-specific examples
    python fact_checker.py --list-domains
        """
    )

    parser.add_argument("input_file", nargs='?', help="Input CSV file path")
    parser.add_argument("output_file", nargs='?', help="Output CSV file path")
    parser.add_argument("--claim-column", "-c", default="claim",
                       help="Name of the column containing claims/facts (default: 'claim')")
    parser.add_argument("--model", "-m", default="VityaVitalich/Llama3.1-8b-instruct",
                       help="Model name/path (default: VityaVitalich/Llama3.1-8b-instruct")
    parser.add_argument("--num-examples", "-n", type=int, default=10,
                       help="Number of few-shot examples (0-10, default: 10)")
    parser.add_argument("--domain", "-d", choices=["medical", "technology", "sports", None],
                       help="Use domain-specific examples (optional)")
    parser.add_argument("--include-confidence", action="store_true",
                       help="Include confidence levels (HIGH/MEDIUM/LOW) in output")
    parser.add_argument("--filtered-data", action="store_true",
                       help="Load only filtered (with is_good flag in csv)")
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=1,
                       help="Number of GPUs for tensor parallelism (default: 2)")
    parser.add_argument("--gpu-memory", type=float, default=0.94,
                       help="GPU memory utilization (default: 0.94)")
    parser.add_argument("--max-model-len", type=int, default=2500,
                       help="Maximum model context length (default: 2500)")
    parser.add_argument("--batch-size", "-b", type=int, default=None,
                       help="Process in batches of this size (default: process all at once)")
    parser.add_argument("--temperature", "-t", type=float, default=0.3,
                       help="Sampling temperature (default: 0.3)")
    parser.add_argument("--max-tokens", type=int, default=150,
                       help="Maximum tokens to generate (default: 150)")
    parser.add_argument("--cache-dir", default="/mnt/extremessd10tb/seleznev/.cache",
                       help="Cache directory for models (default: /mnt/extremessd10tb/seleznev/.cache)")
    parser.add_argument("--list-examples", action="store_true",
                       help="List all available few-shot examples and exit")
    parser.add_argument("--list-domains", action="store_true",
                       help="List all available domain-specific examples and exit")
    
    args = parser.parse_args()
    
    # Handle --list-examples flag
    if args.list_examples:
        print("\nAvailable General Few-Shot Examples:")
        print("="*80)
        examples = get_few_shot_examples()
        for i, ex in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"  Input: {ex['user']}")
            print(f"  Output: {ex['assistant']}")
        print("\n" + "="*80)
        print(f"Total general examples available: {len(examples)}")
        print(f"Use --num-examples N to control how many examples to use (0-{len(examples)})")
        sys.exit(0)
    
    # Handle --list-domains flag
    if args.list_domains:
        print("\nAvailable Domain-Specific Examples:")
        print("="*80)
        domain_examples = get_classification_examples()
        for domain, examples in domain_examples.items():
            print(f"\nDomain: {domain.upper()}")
            print("-" * 40)
            for i, ex in enumerate(examples, 1):
                print(f"  Example {i}:")
                print(f"    Input: {ex['user']}")
                print(f"    Output: {ex['assistant']}")
        print("\n" + "="*80)
        print(f"Available domains: {', '.join(domain_examples.keys())}")
        print("Use --domain DOMAIN_NAME to include domain-specific examples")
        sys.exit(0)
    
    # Validate required arguments
    if not args.input_file or not args.output_file:
        parser.error("Input and output files are required unless using --list-examples or --list-domains")
    
    # Validate num_examples range
    if args.num_examples < 0 or args.num_examples > 10:
        print(f"Error: --num-examples must be between 0 and 10 (got {args.num_examples})")
        sys.exit(1)
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Run fact-checking
    fact_check(
        input_file=args.input_file,
        output_file=args.output_file,
        claim_column=args.claim_column,
        model_name=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_examples=args.num_examples,
        domain=args.domain,
        filtered=args.filtered_data,
        include_confidence=args.include_confidence,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_util=args.gpu_memory,
        max_model_len=args.max_model_len,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
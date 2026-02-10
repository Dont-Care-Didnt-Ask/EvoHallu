"""
Test script to verify attention alignment between scoring and generation modes.

This script compares:
1. Statistics from actual generation (with lm-polygraph's generation)
2. Statistics from scoring pre-existing generations (with our implementation)

Expected result: RAUQ scores should match exactly (within floating point tolerance).
"""

import argparse
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.estimators import MaximumSequenceProbability, Perplexity
from lm_polygraph.estimators.rauq import RAUQ
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

from src.packing.evaluate.uncertainty_quantification.eval_prr_polygraph import (
    compute_stats_for_existing_generation,
    compute_stats_via_generation,
)


def compare_stats(stats1, stats2, name1="Stats1", name2="Stats2"):
    """Compare two statistics dictionaries and report differences."""
    print(f"\n{'=' * 80}")
    print(f"Comparing {name1} vs {name2}")
    print('=' * 80)

    # Compare log likelihoods
    ll1 = stats1["greedy_log_likelihoods"]
    ll2 = stats2["greedy_log_likelihoods"]

    print(f"\n{name1} log likelihoods (first 3 samples, first 5 tokens):")
    for i in range(min(3, len(ll1))):
        print(f"  Sample {i}: {ll1[i][:5]}")

    print(f"\n{name2} log likelihoods (first 3 samples, first 5 tokens):")
    for i in range(min(3, len(ll2))):
        print(f"  Sample {i}: {ll2[i][:5]}")

    # Compute max absolute difference in log likelihoods
    max_ll_diff = 0.0
    for i in range(len(ll1)):
        # Handle different lengths (e.g., if EOS token causes early termination)
        min_len = min(len(ll1[i]), len(ll2[i]))
        for j in range(min_len):
            diff = abs(ll1[i][j] - ll2[i][j])
            max_ll_diff = max(max_ll_diff, diff)

        if len(ll1[i]) != len(ll2[i]):
            print(f"  WARNING: Sample {i} has different lengths: {len(ll1[i])} vs {len(ll2[i])}")

    print(f"\nMax log likelihood difference: {max_ll_diff:.2e}")
    if max_ll_diff < 1e-4:
        print("✓ Log likelihoods MATCH (within tolerance)")
    else:
        print("✗ Log likelihoods DIFFER significantly")

    # Compare attention if present
    if "attention_all" in stats1 and "attention_all" in stats2:
        attn1 = stats1["attention_all"]
        attn2 = stats2["attention_all"]

        print(f"\n{name1} attention shape (first sample): {attn1[0].shape}")
        print(f"{name2} attention shape (first sample): {attn2[0].shape}")

        # Check a few attention values
        if attn1[0].shape[1] >= 2 and attn2[0].shape[1] >= 2:
            print(f"\n{name1} attention [sample 0, layer*head 0, position 1->0]: {attn1[0][0, 1, 0]:.6f}")
            print(f"{name2} attention [sample 0, layer*head 0, position 1->0]: {attn2[0][0, 1, 0]:.6f}")

        max_attn_diff = 0.0
        for i in range(len(attn1)):
            if attn1[i].shape != attn2[i].shape:
                print(f"  WARNING: Sample {i} has different attention shapes: {attn1[i].shape} vs {attn2[i].shape}")
                # Compare overlapping region
                min_dim = min(attn1[i].shape[1], attn2[i].shape[1])
                diff = np.abs(attn1[i][:, :min_dim, :min_dim] - attn2[i][:, :min_dim, :min_dim]).max()
            else:
                diff = np.abs(attn1[i] - attn2[i]).max()
            max_attn_diff = max(max_attn_diff, diff)

        print(f"\nMax attention difference: {max_attn_diff:.2e}")
        if max_attn_diff < 1e-4:
            print("✓ Attention values MATCH (within tolerance)")
        else:
            print("✗ Attention values DIFFER significantly")

    return max_ll_diff


def test_alignment(
    model_id="VityaVitalich/Llama3.1-8b-instruct",
    instruct=True,
    num_samples=5,
    max_new_tokens=20,
):
    """Test alignment between scoring and generation."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Test inputs
    inputs = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "Write a haiku about coding.",
        "What is 2 + 2?",
        "Name three colors.",
    ][:num_samples]

    logging.info(f"Loading model {model_id}...")

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model.eval()

    # Create lm-polygraph WhiteboxModel
    gen_params = GenerationParameters(do_sample=False, max_new_tokens=max_new_tokens)
    model = WhiteboxModel(
        base_model,
        tokenizer,
        model_path=model_id,
        generation_parameters=gen_params,
        instruct=instruct,
    )

    # 1. Generate and compute stats via generation (one sample at a time to avoid padding)
    logging.info("\n" + "=" * 80)
    logging.info("Step 1: Computing stats via GENERATION (individual samples)")
    logging.info("=" * 80)

    # Generate each sample individually to avoid batching/padding artifacts
    all_gen_stats = {
        "greedy_log_likelihoods": [],
        "greedy_tokens": [],
        "greedy_texts": [],
        "greedy_log_probs": [],
        "attention_all": [],
        "model": model,
    }

    for i, inp in enumerate(inputs):
        logging.info(f"Generating sample {i+1}/{len(inputs)}...")
        sample_stats = compute_stats_via_generation(
            model=model,
            input_texts=[inp],  # Single input
            max_new_tokens=max_new_tokens,
            output_attentions=True,
        )

        # Accumulate stats
        all_gen_stats["greedy_log_likelihoods"].extend(sample_stats["greedy_log_likelihoods"])
        all_gen_stats["greedy_tokens"].extend(sample_stats["greedy_tokens"])
        all_gen_stats["greedy_texts"].extend(sample_stats["greedy_texts"])
        all_gen_stats["greedy_log_probs"].extend(sample_stats["greedy_log_probs"])
        all_gen_stats["attention_all"].extend(sample_stats["attention_all"])

    stats_gen = all_gen_stats

    logging.info(f"Generated {len(stats_gen['greedy_texts'])} samples")
    logging.info("Example generations:")
    for i in range(min(3, len(stats_gen["greedy_texts"]))):
        text = stats_gen['greedy_texts'][i]
        tokens = stats_gen['greedy_tokens'][i]
        logging.info(f"  {i}: {text[:80]}...")
        logging.info(f"     Tokens: {len(tokens)}, Has EOS: {tokenizer.eos_token_id in tokens}")

    # 2. Use those generations to compute stats via scoring
    # For exact comparison, we need to append EOS to texts that had EOS in generation
    logging.info("\n" + "=" * 80)
    logging.info("Step 2: Computing stats via SCORING (using generated texts)")
    logging.info("=" * 80)

    # Append EOS token representation to texts that had EOS in generation
    generation_texts_for_scoring = []
    for i, text in enumerate(stats_gen["greedy_texts"]):
        if tokenizer.eos_token_id in stats_gen["greedy_tokens"][i]:
            # Generation included EOS, append it to the text
            # EOS tokens are typically not in decoded text, so we append the actual token
            text_with_eos = text + tokenizer.decode([tokenizer.eos_token_id])
            generation_texts_for_scoring.append(text_with_eos)
        else:
            # Generation didn't include EOS (hit max_tokens), use text as-is
            generation_texts_for_scoring.append(text)

    stats_score = compute_stats_for_existing_generation(
        model=model,
        input_texts=inputs,
        generation_texts=generation_texts_for_scoring,
        output_attentions=True,
    )

    logging.info("Scoring token lengths:")
    for i in range(min(3, len(stats_score["greedy_tokens"]))):
        tokens = stats_score['greedy_tokens'][i]
        logging.info(f"  {i}: Tokens: {len(tokens)}, Has EOS: {tokenizer.eos_token_id in tokens}")

    # 3. Compare statistics
    max_ll_diff = compare_stats(stats_gen, stats_score, "Generation", "Scoring")

    # 4. Test estimators
    print(f"\n{'=' * 80}")
    print("Testing Estimators")
    print('=' * 80)

    estimators = [
        ("MaximumSequenceProbability", MaximumSequenceProbability()),
        ("Perplexity", Perplexity()),
        ("RAUQ", RAUQ(instruct=instruct)),
    ]

    results = {}
    for name, estimator in estimators:
        try:
            scores_gen = estimator(stats_gen)
            scores_score = estimator(stats_score)

            if not isinstance(scores_gen, np.ndarray):
                scores_gen = np.array(scores_gen)
            if not isinstance(scores_score, np.ndarray):
                scores_score = np.array(scores_score)

            max_diff = np.abs(scores_gen - scores_score).max()
            mean_diff = np.abs(scores_gen - scores_score).mean()

            print(f"\n{name}:")
            print(f"  Generation scores:  {scores_gen}")
            print(f"  Scoring scores:     {scores_score}")
            print(f"  Max difference:     {max_diff:.2e}")
            print(f"  Mean difference:    {mean_diff:.2e}")

            if max_diff < 1e-3:
                print(f"  ✓ {name} scores MATCH")
                results[name] = "PASS"
            else:
                print(f"  ✗ {name} scores DIFFER")
                results[name] = "FAIL"

        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results[name] = "ERROR"

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)
    print(f"Max log likelihood difference: {max_ll_diff:.2e}")
    for name, status in results.items():
        print(f"{name:40s}: {status}")

    all_passed = all(status == "PASS" for status in results.values())
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        return True
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test attention alignment between scoring and generation"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="VityaVitalich/Llama3.1-8b-instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        default=True,
        help="Use instruction-tuned model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test samples",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Max tokens to generate",
    )

    args = parser.parse_args()

    success = test_alignment(
        model_id=args.model_id,
        instruct=args.instruct,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

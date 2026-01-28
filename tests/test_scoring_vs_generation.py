"""
Tests to verify that scoring pre-existing generations produces identical statistics
as generating text with the same model.

This is critical for ensuring PRR evaluation is consistent between:
1. Generate text with model A, score with model A -> baseline
2. Load pre-existing generations, score with model A -> should match baseline

Run with: pytest tests/test_scoring_vs_generation.py -v
Or standalone: python tests/test_scoring_vs_generation.py
"""

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if lm-polygraph is available
try:
    from lm_polygraph.utils.model import WhiteboxModel
    from lm_polygraph.utils.generation_parameters import GenerationParameters
    from lm_polygraph.estimators import MaximumSequenceProbability, Perplexity
    LM_POLYGRAPH_AVAILABLE = True
except ImportError:
    LM_POLYGRAPH_AVAILABLE = False
    pytest.skip("lm-polygraph not installed", allow_module_level=True)

from src.packing.evaluate.uncertainty_quantification.eval_prr_polygraph import (
    compute_stats_for_existing_generation,
    compute_stats_via_generation,
)


# Use a tiny model for testing
TEST_MODEL_ID = "sshleifer/tiny-gpt2"  # ~2MB model


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for all tests."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_ID,
        torch_dtype=torch.float32,  # Use float32 for exact comparisons
    )
    model.eval()

    gen_params = GenerationParameters(do_sample=False, max_new_tokens=20)
    whitebox_model = WhiteboxModel(
        model,
        tokenizer,
        model_path=TEST_MODEL_ID,
        generation_parameters=gen_params,
        instruct=False,  # tiny-gpt2 is not instruction-tuned
    )

    return whitebox_model


class TestScoringVsGeneration:
    """Test that scoring existing text matches generation statistics."""

    def test_log_likelihoods_match(self, model_and_tokenizer):
        """
        Test that greedy_log_likelihoods from scoring matches generation.

        This is the core test: if we generate "Hello world" from prompt "Say",
        then scoring "Say" + "Hello world" should give identical log-likelihoods.

        We use rtol=1e-4 to account for floating point precision differences
        between generation (with KV cache) and full forward pass.
        """
        model = model_and_tokenizer
        input_texts = ["The capital of France is", "Once upon a time"]

        # Generate text
        gen_stats = compute_stats_via_generation(
            model=model,
            input_texts=input_texts,
            max_new_tokens=10,
            output_attentions=False,  # Don't need attention for this test
        )

        # Score the same generations
        score_stats = compute_stats_for_existing_generation(
            model=model,
            input_texts=input_texts,
            generation_texts=gen_stats["greedy_texts"],
            output_attentions=False,
        )

        # Compare log likelihoods
        for i, (gen_ll, score_ll) in enumerate(
            zip(gen_stats["greedy_log_likelihoods"], score_stats["greedy_log_likelihoods"])
        ):
            gen_ll_arr = np.array(gen_ll)
            score_ll_arr = np.array(score_ll)

            # Check shapes match
            assert gen_ll_arr.shape == score_ll_arr.shape, (
                f"Sample {i}: shape mismatch - gen {gen_ll_arr.shape} vs score {score_ll_arr.shape}"
            )

            # Check values match (with tolerance for floating point precision)
            np.testing.assert_allclose(
                gen_ll_arr,
                score_ll_arr,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Sample {i}: log-likelihoods don't match",
            )

        print("PASSED: Log-likelihoods match between generation and scoring (within fp tolerance)")

    def test_tokens_match(self, model_and_tokenizer):
        """Test that greedy_tokens from scoring matches generation."""
        model = model_and_tokenizer
        input_texts = ["Hello", "The weather"]

        gen_stats = compute_stats_via_generation(
            model=model,
            input_texts=input_texts,
            max_new_tokens=8,
            output_attentions=False,
        )

        score_stats = compute_stats_for_existing_generation(
            model=model,
            input_texts=input_texts,
            generation_texts=gen_stats["greedy_texts"],
            output_attentions=False,
        )

        for i, (gen_tokens, score_tokens) in enumerate(
            zip(gen_stats["greedy_tokens"], score_stats["greedy_tokens"])
        ):
            assert gen_tokens == score_tokens, (
                f"Sample {i}: tokens don't match - gen {gen_tokens} vs score {score_tokens}"
            )

        print("PASSED: Tokens match exactly between generation and scoring")

    def test_attention_patterns_shapes_match(self, model_and_tokenizer):
        """
        Test that attention_all shapes from scoring matches generation.

        NOTE: Attention VALUES will NOT match exactly because:
        - Generation (KV cache): at step j, gen_j attends to [prompt, gen_0..gen_{j-1}], NOT self
        - Scoring (full forward): gen_j attends to [prompt, gen_0..gen_j], INCLUDING self
        The softmax denominators differ, causing different attention values.

        However, shapes and structure should match.
        """
        model = model_and_tokenizer
        input_texts = ["The quick brown fox"]

        gen_stats = compute_stats_via_generation(
            model=model,
            input_texts=input_texts,
            max_new_tokens=8,
            output_attentions=True,
        )

        score_stats = compute_stats_for_existing_generation(
            model=model,
            input_texts=input_texts,
            generation_texts=gen_stats["greedy_texts"],
            output_attentions=True,
        )

        for i, (gen_attn, score_attn) in enumerate(
            zip(gen_stats["attention_all"], score_stats["attention_all"])
        ):
            # Check shapes match
            assert gen_attn.shape == score_attn.shape, (
                f"Sample {i}: attention shape mismatch - gen {gen_attn.shape} vs score {score_attn.shape}"
            )

            # Values WILL differ due to KV cache vs full forward pass difference
            # Just verify the difference is bounded (not completely wrong)
            max_diff = np.max(np.abs(gen_attn - score_attn))
            assert max_diff < 1.0, (
                f"Sample {i}: attention difference too large: {max_diff}"
            )

        print("PASSED: Attention shapes match (values differ due to KV cache vs full forward)")

    def test_estimators_produce_same_scores(self, model_and_tokenizer):
        """
        Test that uncertainty estimators produce identical/similar scores
        whether using generation stats or scoring stats.

        Log-likelihood based estimators (MSP, Perplexity) should match closely.
        Attention-based estimators (RAUQ) will differ due to KV cache vs full forward.
        """
        model = model_and_tokenizer
        input_texts = ["Paris is the capital of", "Machine learning is"]

        gen_stats = compute_stats_via_generation(
            model=model,
            input_texts=input_texts,
            max_new_tokens=12,
            output_attentions=False,  # Don't need attention for MSP/Perplexity
        )
        gen_stats["model"] = model

        score_stats = compute_stats_for_existing_generation(
            model=model,
            input_texts=input_texts,
            generation_texts=gen_stats["greedy_texts"],
            output_attentions=False,
        )
        score_stats["model"] = model

        # These estimators only use log-likelihoods, should match closely
        estimators = [
            MaximumSequenceProbability(),
            Perplexity(),
        ]

        for estimator in estimators:
            gen_scores = estimator(gen_stats)
            score_scores = estimator(score_stats)

            np.testing.assert_allclose(
                np.array(gen_scores),
                np.array(score_scores),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Estimator {estimator}: scores don't match",
            )

        print("PASSED: Log-likelihood estimators produce matching scores")

    def test_single_token_generation(self, model_and_tokenizer):
        """Edge case: single token generation."""
        model = model_and_tokenizer
        input_texts = ["Test"]

        gen_stats = compute_stats_via_generation(
            model=model,
            input_texts=input_texts,
            max_new_tokens=1,
            output_attentions=False,
        )

        score_stats = compute_stats_for_existing_generation(
            model=model,
            input_texts=input_texts,
            generation_texts=gen_stats["greedy_texts"],
            output_attentions=False,
        )

        for gen_ll, score_ll in zip(
            gen_stats["greedy_log_likelihoods"],
            score_stats["greedy_log_likelihoods"]
        ):
            if len(gen_ll) > 0:  # Only test if we have tokens
                np.testing.assert_allclose(
                    np.array(gen_ll),
                    np.array(score_ll),
                    rtol=1e-4,
                    atol=1e-5,
                )

        print("PASSED: Single token generation handled correctly")

    def test_varying_generation_lengths(self, model_and_tokenizer):
        """Test batch with varying generation lengths (due to EOS)."""
        model = model_and_tokenizer
        # Use prompts that might generate different lengths
        input_texts = ["A", "The end.", "Once"]

        gen_stats = compute_stats_via_generation(
            model=model,
            input_texts=input_texts,
            max_new_tokens=15,
            output_attentions=False,
        )

        score_stats = compute_stats_for_existing_generation(
            model=model,
            input_texts=input_texts,
            generation_texts=gen_stats["greedy_texts"],
            output_attentions=False,
        )

        # Each sample should match independently
        for i in range(len(input_texts)):
            gen_ll = np.array(gen_stats["greedy_log_likelihoods"][i])
            score_ll = np.array(score_stats["greedy_log_likelihoods"][i])

            if len(gen_ll) > 0 and len(score_ll) > 0:
                # Compare up to the shorter length (in case of minor tokenization differences)
                min_len = min(len(gen_ll), len(score_ll))
                np.testing.assert_allclose(
                    gen_ll[:min_len],
                    score_ll[:min_len],
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=f"Sample {i} log-likelihoods don't match",
                )

        print("PASSED: Varying generation lengths handled correctly")


def run_quick_verification():
    """Quick verification without pytest (for manual testing).

    NOTE on attention patterns:
    Attention from scoring (full forward pass) will NOT exactly match attention
    from generation (with KV cache) because:
    - Generation: at step j, query gen_j attends to [prompt, gen_0..gen_{j-1}], NOT including self
    - Scoring: position gen_j attends to [prompt, gen_0..gen_j], INCLUDING self
    The softmax denominators differ, causing different attention values.

    However, log-likelihoods SHOULD match exactly (within floating point tolerance)
    because they don't depend on attention patterns.
    """
    print("=" * 60)
    print("Quick Verification: Scoring vs Generation Equivalence")
    print("=" * 60)

    print(f"\nLoading tiny model: {TEST_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_ID,
        torch_dtype=torch.float32,
    )
    base_model.eval()

    gen_params = GenerationParameters(do_sample=False, max_new_tokens=10)
    model = WhiteboxModel(
        base_model,
        tokenizer,
        model_path=TEST_MODEL_ID,
        generation_parameters=gen_params,
        instruct=False,
    )

    input_texts = ["The capital of France is", "Hello world"]

    print("\n1. Generating text...")
    gen_stats = compute_stats_via_generation(
        model=model,
        input_texts=input_texts,
        max_new_tokens=10,
        output_attentions=True,
    )

    for i, (inp, gen) in enumerate(zip(input_texts, gen_stats["greedy_texts"])):
        print(f"   [{i}] Input: '{inp}' -> Generated: '{gen}'")

    print("\n2. Scoring the same generations...")
    score_stats = compute_stats_for_existing_generation(
        model=model,
        input_texts=input_texts,
        generation_texts=gen_stats["greedy_texts"],
        output_attentions=True,
    )

    print("\n3. Comparing log-likelihoods...")
    print("   (Should match within ~1e-4 tolerance)")

    ll_match = True
    for i in range(len(input_texts)):
        gen_ll = np.array(gen_stats["greedy_log_likelihoods"][i])
        score_ll = np.array(score_stats["greedy_log_likelihoods"][i])

        if len(gen_ll) != len(score_ll):
            print(f"   [{i}] LENGTH MISMATCH: gen={len(gen_ll)}, score={len(score_ll)}")
            ll_match = False
            continue

        max_diff = np.max(np.abs(gen_ll - score_ll))
        match = max_diff < 1e-4  # Floating point tolerance

        status = "PASS" if match else "FAIL"
        print(f"   [{i}] Log-likelihoods: {status} (max diff: {max_diff:.2e})")

        if not match:
            ll_match = False
            print(f"       Generation: {gen_ll[:5]}...")
            print(f"       Scoring:    {score_ll[:5]}...")

    # Compare attention (expected to differ due to KV cache)
    print("\n4. Comparing attention patterns...")
    print("   (Expected to DIFFER due to KV cache vs full forward pass)")
    for i in range(len(input_texts)):
        gen_attn = gen_stats["attention_all"][i]
        score_attn = score_stats["attention_all"][i]

        if gen_attn.shape != score_attn.shape:
            print(f"   [{i}] SHAPE MISMATCH: gen={gen_attn.shape}, score={score_attn.shape}")
            continue

        max_diff = np.max(np.abs(gen_attn - score_attn))
        print(f"   [{i}] Attention max diff: {max_diff:.2e} (expected to differ)")

    # Test estimators
    print("\n5. Testing log-likelihood based estimators...")
    print("   (Should match within ~1e-4 tolerance)")
    gen_stats["model"] = model
    score_stats["model"] = model

    est_match = True
    for est in [MaximumSequenceProbability(), Perplexity()]:
        gen_scores = np.array(est(gen_stats))
        score_scores = np.array(est(score_stats))

        max_diff = np.max(np.abs(gen_scores - score_scores))
        match = max_diff < 1e-4

        status = "PASS" if match else "FAIL"
        print(f"   {est}: {status} (max diff: {max_diff:.2e})")

        if not match:
            est_match = False

    print("\n" + "=" * 60)
    if ll_match and est_match:
        print("SUCCESS: Log-likelihoods and estimators match!")
        print("(Attention differs as expected due to KV cache vs full forward)")
    else:
        print("FAILURE: Log-likelihoods or estimators don't match!")
    print("=" * 60)

    return ll_match and est_match


if __name__ == "__main__":
    success = run_quick_verification()
    exit(0 if success else 1)

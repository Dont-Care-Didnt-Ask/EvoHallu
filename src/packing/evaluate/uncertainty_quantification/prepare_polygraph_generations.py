"""Script to obtain model generations for LM-Polygraph benchmark datasets.

For each dataset, saves a CSV file with columns: "input", "model_generation", "output".
Uses greedy decoding with configurable max tokens.
"""

import argparse
import logging
from pathlib import Path

import datasets
import pandas as pd
import torch
import transformers
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# (dataset_name, config) - use simple_instruct where available, continuation otherwise
DATASETS = [
    ("LM-Polygraph/wmt14", "deen"),
    ("LM-Polygraph/wmt14", "fren"),
    ("LM-Polygraph/wmt19", "deen_simple_instruct"),
    ("LM-Polygraph/wmt19", "ruen_simple_instruct"),
    # ("LM-Polygraph/aeslc", "continuation"),
    ("LM-Polygraph/coqa", "simple_instruct"),
    ("LM-Polygraph/babi_qa", "continuation"),
    ("LM-Polygraph/gsm8k", "simple_instruct"),
    ("LM-Polygraph/triviaqa", "simple_instruct"),
    ("LM-Polygraph/truthfulqa", "simple_instruct"),
    ("LM-Polygraph/mmlu", "simple_instruct"),
    ("LM-Polygraph/samsum", "simple_instruct"),
    ("LM-Polygraph/xsum", "simple_instruct"),
]

def get_output_path(output_dir: Path, dataset_name: str, config: str, limit: int | None = None) -> Path:
    """Get the output CSV path for a dataset."""
    # Convert ("LM-Polygraph/wmt14", "deen") -> "wmt14_deen.csv"
    # Or with limit: "wmt14_deen_limit_5.csv"
    base_name = dataset_name.split("/")[-1]
    if limit is not None:
        filename = f"{base_name}_{config}_limit_{limit}.csv"
    else:
        filename = f"{base_name}_{config}.csv"
    return output_dir / filename


def get_partial_output_path(output_dir: Path, dataset_name: str, config: str, limit: int | None = None) -> Path:
    """Get the partial (in-progress) CSV path for a dataset."""
    base_name = dataset_name.split("/")[-1]
    if limit is not None:
        filename = f"PARTIAL_{base_name}_{config}_limit_{limit}.csv"
    else:
        filename = f"PARTIAL_{base_name}_{config}.csv"
    return output_dir / filename


def load_model_and_tokenizer(
    model_name: str, dtype: torch.dtype
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Load the model and tokenizer."""
    logger.info(f"Loading model: {model_name}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Set padding side to left for decoder-only models (required for batched generation)
    tokenizer.padding_side = "left"

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Model loaded on device(s): {model.hf_device_map}")
    return model, tokenizer


def format_inputs(
    inputs: list[str],
    tokenizer: transformers.PreTrainedTokenizer,
    apply_chat_template: bool,
) -> list[str]:
    """Optionally apply chat template to inputs."""
    if not apply_chat_template:
        return inputs

    formatted = []
    for text in inputs:
        messages = [{"role": "user", "content": text}]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        formatted.append(formatted_text)
    return formatted


def generate_batch(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Generate responses for a batch of prompts using greedy decoding."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (exclude input tokens)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_length:]
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return generations


def process_dataset(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_name: str,
    config: str,
    output_path: Path,
    partial_output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    apply_chat_template: bool,
    limit: int | None,
) -> None:
    """Process a single dataset and save generations to CSV.

    Writes to partial_output_path during generation, then atomically renames
    to output_path when complete to ensure data integrity.
    """
    logger.info(f"Loading dataset: {dataset_name} (config: {config})")
    # Bypass verification for triviaqa due to deprecated 'List' feature type in metadata
    if "triviaqa" in dataset_name.lower():
        dataset = datasets.load_dataset(
            dataset_name,
            config,
            split="test",
            trust_remote_code=True,
            verification_mode="no_checks"
        )
    else:
        dataset = datasets.load_dataset(dataset_name, config, split="test")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
        logger.info(f"Limited to {len(dataset)} examples")

    inputs = dataset["input"]
    outputs = dataset["output"]

    # Format inputs if chat template is requested
    formatted_inputs = format_inputs(inputs, tokenizer, apply_chat_template)

    # Generate in batches and write incrementally to PARTIAL file to avoid memory issues
    num_batches = (len(formatted_inputs) + batch_size - 1) // batch_size

    # Write header to partial file
    df_header = pd.DataFrame(columns=["input", "model_generation", "output"])
    df_header.to_csv(partial_output_path, index=False, mode='w')

    total_saved = 0
    for i in tqdm(range(0, len(formatted_inputs), batch_size), total=num_batches, desc=dataset_name):
        batch_prompts = formatted_inputs[i : i + batch_size]
        batch_generations = generate_batch(model, tokenizer, batch_prompts, max_new_tokens)

        # Create DataFrame for this batch and append to CSV
        batch_size_actual = len(batch_generations)
        df_batch = pd.DataFrame(
            {
                "input": inputs[i : i + batch_size_actual],
                "model_generation": batch_generations,
                "output": outputs[i : i + batch_size_actual],
            }
        )
        df_batch.to_csv(partial_output_path, index=False, mode='a', header=False)
        total_saved += len(df_batch)

    # Atomically rename partial file to final file (ensures complete dataset or nothing)
    partial_output_path.rename(output_path)
    logger.info(f"Saved {total_saved} generations to {output_path}")


def main() -> None:
    args = parse_args()

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, dtype)

    # Process each dataset
    for dataset_name, config in DATASETS:
        output_path = get_output_path(output_dir, dataset_name, config, limit=args.limit)
        partial_output_path = get_partial_output_path(output_dir, dataset_name, config, limit=args.limit)

        if output_path.exists() and not args.overwrite:
            logger.info(f"Skipping {dataset_name}/{config}: output file already exists at {output_path}")
            continue

        # Clean up any leftover partial file from previous failed run for this dataset
        if partial_output_path.exists():
            logger.warning(f"Removing incomplete partial file from previous run: {partial_output_path}")
            partial_output_path.unlink()

        logger.info(f"Processing {dataset_name}/{config}")
        process_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            config=config,
            output_path=output_path,
            partial_output_path=partial_output_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            apply_chat_template=not args.no_chat_template,
            limit=args.limit,
        )

    logger.info("All datasets processed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model outputs for LM-Polygraph benchmark datasets."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="VityaVitalich/Llama3.1-8b-instruct",
        help="HuggingFace model name or path (e.g., 'VityaVitalich/Llama3.1-8b-instruct').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/uncertainty_quantification/polygraph_generations",
        help="Directory to save output CSV files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate (default: 2048).",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template (by default, chat template is applied).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. If not set, existing files are skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per dataset (useful for debugging, e.g., --limit 23).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()

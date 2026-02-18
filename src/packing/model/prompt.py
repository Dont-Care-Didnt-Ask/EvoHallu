import numpy as np
import wandb
import copy

from packing.evaluate.registry import TASK_REGISTRY

# Keys to exclude from the additional metrics display (e.g., long text fields)
_EXCLUDE_METRIC_KEYS = {
    "prompt_text",
    "AUC 95% confidence interval (DeLong)",
    "Precision at median score threshold",
    "Recall at median score threshold",
}


def _format_metric_value(value):
    """Format a metric value for display: floats with 4 decimals, others as-is."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_extra_metrics(em: dict) -> str:
    """Format extra_metrics dict into a readable string, excluding certain keys."""
    
    # Multi-criteria optimization task variant
    if "datasets" in em:
        dataset_info = []
        for dataset_name in em["datasets"].keys():
            results = em["datasets"][dataset_name]
            prr = f"{100 * results["prediction_rejection_ratio"]:.1f}"
            dataset_info.append(f"# Dataset: {dataset_name}\tPRR: {prr}")
        
        return "# === Per-dataset metrics ===\n" + "\n".join(dataset_info) + "\n# ======\n"
    
    # Single-criteria optimization task variant
    parts = [
        f"{k}: {_format_metric_value(v)}"
        for k, v in em.items()
        if k not in _EXCLUDE_METRIC_KEYS
    ]
    metrics_str = ", ".join(parts)
    return f"# Additional metrics: {metrics_str}\n"  if parts else ""


def programs_to_prompt_creative(cfg, programs, scores, extra_metrics_list):
    """
    Given a list of programs, their scores, and extra_metrics, construct a prompt that encourages creative solutions.
    """
    main_txt = ""
    for program, score, em in zip(programs, scores, extra_metrics_list):
        main_txt += f"{program}\n"
    
        if "task_variant" in em and em["task_variant"] == "multi-criteria optimization":
            main_txt += f"# Score achieved with the function above (mean + min PRR / 2): {score}\n"
        else:
            main_txt += f"# Score achieved with the function above: {score}\n"

        if em:
            main_txt += _format_extra_metrics(em)
        main_txt += "\n"

    task = TASK_REGISTRY.get(cfg.task.task_name)
    system_prompt = task["system_prompt"]
    append_prompt = task["append_prompt"]

    chat = [
        {"role": "system", "content": system_prompt if isinstance(system_prompt, str) else system_prompt()},
        {
            "role": "user",
            "content": f"{main_txt}\n{append_prompt}",
        },
    ]
    return chat, scores


def programs_to_prompt(cfg, programs, scores, extra_metrics_list):
    """
    Given a list of programs, their scores, and extra_metrics, construct a prompt.
    """

    if cfg.descending_order:
        programs = programs[::-1]
        scores = scores[::-1]
        extra_metrics_list = extra_metrics_list[::-1]

    code_type = getattr(cfg, "code_type", "function")
    code_desc = "class" if code_type == "class" else "function"

    main_txt = ""
    for program, score, em in zip(programs, scores, extra_metrics_list):
        main_txt += f"{program}\n"
        main_txt += f"# Score achieved with the {code_desc} above: {score}\n"
        if em:
            metrics_str = _format_extra_metrics(em)
            if metrics_str:
                main_txt += f"# Additional metrics: {metrics_str}\n"
        main_txt += "\n"

    if code_type == "class":
        user_content = f"{main_txt}\nWrite a new class {cfg.function_str_to_extract} that will perform better than the classes above and achieve a higher score."
    else:
        user_content = f"{main_txt}\nWrite a new function {cfg.function_str_to_extract}() that will perform better than both functions above and achieve a higher score."

    chat = [
        {
            "role": "user",
            "content": user_content,
        }
    ]
    return chat, scores


def generate_prompt(cfg, programdatabase, current_percentile):
    """
    Generates a single prompt.
    """
    # Sample from the programbank
    programs, scores, island_id_prompt, temperature, probabilities, extra_metrics_list = programdatabase.get_prompt(percentile=current_percentile)

    # Construct prompt
    if cfg.creative_prompt:
        chat, prompt_scores = programs_to_prompt_creative(cfg, programs, scores, extra_metrics_list)
    else:
        chat, prompt_scores = programs_to_prompt(cfg, programs, scores, extra_metrics_list)
    return chat, prompt_scores, island_id_prompt, temperature, probabilities


def generate_batch_prompts(cfg, programdatabase, running_dict, round_num):
    """
    Generates cfg.num_cont_rounds prompts that will be used by the producer. 
    """
    chats_batch = []
    island_id_prompt_batch = []
    prompt_scores_batch = []
    temperatures_batch = []
    probabilities_batch = []
    prompt_nums_batch = []
    extra_metrics_batch = []
    # Linear decrease in the percentile based on how far we are in the rounds
    initial_percentile = cfg.initial_percentile
    final_percentile = cfg.final_percentile
    current_percentile = initial_percentile - (initial_percentile - final_percentile) * round_num / cfg.num_rounds
    running_dict["current_percentile"] = current_percentile

    for _ in range(cfg.num_cont_rounds):
        chat, prompt_scores, island_id_prompt, temperature, probabilities = generate_prompt(
            cfg, programdatabase, current_percentile
        )
        chats_batch.append(chat)
        island_id_prompt_batch.append(island_id_prompt)
        prompt_scores_batch.append(prompt_scores)
        temperatures_batch.append(temperature)
        probabilities_batch.append(probabilities)
        running_dict["prompt_num"] += 1
        prompt_nums_batch.append(copy.deepcopy(running_dict["prompt_num"]))

    return (
        chats_batch,
        island_id_prompt_batch,
        prompt_scores_batch,
        temperatures_batch,
        probabilities_batch,
        prompt_nums_batch,
        running_dict,
    )

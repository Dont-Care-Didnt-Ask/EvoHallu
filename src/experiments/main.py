import os

from packing.evaluate.registry import TASK_REGISTRY

os.environ["JAX_PLATFORM_NAME"] = "cpu"  # We can't use GPU with multiprocessing for evaluation very well yet
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_SILENT"] = "true"

import torch
import numpy as np
import wandb
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging
import csv
import copy
import threading
import multiprocessing
import faulthandler
import traceback
import sys
import time
faulthandler.enable()

from packing.utils.seeding import seed_everything, generate_random_seed
from packing.utils.functions import function_to_string

from packing.funsearch.programs_database import ProgramsDatabase

from hydra import main as hydra_main

from packing.train.dpo.dpo_dataset import DPODataBuffer
from packing.train.dpo.dpo import prepare_dpo_chats
from packing.model.model import (
    initialize_models,
    get_full_model_name,
    delete_sampling_model,
    initialize_models_server,
    kill_process_with_pid_and_wait,
)
from packing.model.prompt import generate_batch_prompts
from packing.logging.logging import (
    register_programs,
    save_func_class_to_file,
    initialize_running_dict,
    get_pd_statistics,
    wandb_log_running_dict,
    calculate_avg_score,
    calculate_avg_passed_score,
    log_train_keys_containing_loop,
)
from packing.logging.function_class import FunctionClass
from packing.logging.log_to_file import (
    save_pd_to_file,
    save_in_logsdir,
    load_from_logsdir,
)
from packing.parallel.continuous_execution import (
    Producer,
    Consumer,
    safe_consumer,
    consumers_finish_and_cleanup,
)

from experiments.hallu_utils import evaluate_hallu, write_test_metrics_row
from experiments.uq_utils import evaluate_uq, write_uq_metrics_row, evaluate_uq_multi, write_uq_multi_metrics_row


OmegaConf.register_new_resolver("generate_random_seed", generate_random_seed, use_cache=True)
OmegaConf.register_new_resolver("eval", eval, use_cache=True)
import pickle
import subprocess
import getpass



@hydra_main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)  # So we can assign new fields

    # Needed since we always store DPO chats
    cfg.train.dpo_strategy = 2 if "dpo_strategy" not in cfg.train else cfg.train.dpo_strategy

    torch.cuda.set_per_process_memory_fraction(0.99, torch.cuda.current_device())

    if cfg.task.task_name in {'flatpack', 'uncertainty_quantification'}:
        # Unfortunately, jax doesn't work with forked multithreading since it is multithreaded itself,
        # so we need to spawn the processes instead
        # uncertainty quantification task uses numpy, which seems to have issues with forked processes also
        multiprocessing.set_start_method('spawn')

    # Perform checks on the config to ensure it is valid
    if not cfg.one_tuning:
        assert cfg.max_loops > 1, "If not one_tuning, max_loops must be greater than 1"
    if cfg.task.task_name == 'bin':
        assert not (cfg.task.Weibull and cfg.task.OR), "Only one bin packing dataset can be selected"

    # Overwrite "overwrite" variables in the config
    cfg.logs_dir = (
        f"out/logs/{cfg.prefix}/{cfg.task.task_name}_{cfg.run_identifier_name}_{cfg.seed}"
    )
    cfg.wandb_name = f"{cfg.prefix}/{cfg.run_identifier_name}_{cfg.seed}"
    cfg.group_name = f"{cfg.prefix}/{cfg.run_identifier_name}"

    # Change the accelerate config in the case of only using one GPU, since we might want to use a specific GPU id
    # This works hard-coded up to 4 GPUs, otherwise just make new accelerate config files in this format
    if cfg.accelerate_config == "1gpu":
        if isinstance(cfg.gpu_nums, int):
            cfg.accelerate_config = cfg.accelerate_config + f'_{cfg.gpu_nums}'

    user = getpass.getuser()
    # Assume you're running from the repository's base dir
    cfg.full_accelerate_config = f"./configs/accelerate_config/{cfg.accelerate_config}.yaml"
    assert cfg.use_tgi + cfg.use_vllm < 2, "Only up to one inference server can be selected"

    # Model specific cfg
    cfg.full_model_name = get_full_model_name(cfg)
    if torch.cuda.get_device_properties(0).major < 8:
        cfg.model_dtype = "float16"
    else:
        cfg.model_dtype = "bfloat16"

    if cfg.multiple_models:
        cfg.model_adapter_dir = [
            f"{cfg.logs_dir}/model_adapter_{model_name}" for model_name in cfg.model_name
        ]
    else:
        cfg.model_adapter_dir = f"{cfg.logs_dir}/model_adapter_{cfg.model.model_name}"

    # Create logs directory if it doesn't exist
    os.makedirs(cfg.logs_dir, exist_ok=True)

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{cfg.logs_dir}/stdout.log")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Task specific imports
    try:
        task = TASK_REGISTRY.get(cfg.task.task_name)
    except ValueError as e:
        raise RuntimeError(
            f"Task '{cfg.task.task_name}' is not registered. Check if it's implemented and registered properly.") from e

    generate_input = task["generate_input"]
    evaluate_func = task["evaluate_func"]
    get_initial_func = task["get_initial_func"]

    initial_function, function_str_to_extract, code_type = get_initial_func(cfg.task)
    cfg.function_str_to_extract = function_str_to_extract
    cfg.code_type = code_type  # "function" or "class"

    # Save the config and set up wandb
    logging.info(f"Working directory: {Path.cwd()}")
    if cfg.wandb:
        default_run = wandb.init(
            project=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.wandb_name,
            group=cfg.group_name,
            reinit=True,
            entity=cfg.entity,
        )
        cfg.default_wandb_name = wandb.run.id
    with open(f"{cfg.logs_dir}/config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    logging.info(f"Running with config: \n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # DO NOT SEED IF USING MULTIPLE COPY OF THE MODEL ON MULTIPLE GPUS, AS ALL GENERATORS WILL BE THE SAME
    seed_everything(cfg.seed)
    if cfg.multiple_models:
        assert len(set(cfg.model.model_name)) == len(
            cfg.model.model_name
        ), "Model names must be unique if using seeding, otherwise we are generating same outputs"

    # Are we resuming from a previous checkpoint or starting from scratch?
    if os.path.exists(f"{cfg.logs_dir}/flag_resume.txt"):
        logging.info("Resuming from logs directory")
        programdatabase, dpo_chats, running_dict, starting_round, input_struct = (
            load_from_logsdir(cfg.logs_dir)
        )
        starting_round += cfg.num_cont_rounds

        logging.info(f"Total functions evaluated: {running_dict['num_func_evaluated']}")
        logging.info(
            f"Number of functions in the programbank: {programdatabase.total_num_programs}"
        )
        logging.info(f"Number of functions in the finetuning data: {len(dpo_chats)}")
    else:
        logging.info("Starting from scratch")
        input_struct = generate_input(cfg.task, "train")
        running_dict = initialize_running_dict(
            cfg, input_struct
        )  # Stores all the running statistics and flags
        programdatabase = ProgramsDatabase(cfg.programdatabaseConfig)
        dpo_chats = DPODataBuffer()
        starting_round = 0

        # Register the intial function into the programbank
        initial_function_str = function_to_string(initial_function)
        intial_function_class = FunctionClass(
            initial_function_str, ""
        )
        intial_results = evaluate_func(cfg, copy.deepcopy(input_struct), intial_function_class)
        programdatabase.register_program(initial_function_str, intial_results.score, None, intial_results.extra_metrics)
        save_func_class_to_file(cfg, intial_results)

        # Evaluate initial prompt on the test set (only for hallu task)
        if cfg.task.task_name == "hallu":
            logging.info("Evaluating initial prompt on test set...")
            for split in ["val", "testset"]:
                initial_test_result, num_test_samples = evaluate_hallu(cfg, generate_input, evaluate_func, intial_function_class, split)
                write_test_metrics_row(cfg, initial_test_result, split, num_test_samples, iteration=-1)
                logging.info(f"Initial prompt {split} evaluation saved to test_metrics.csv")

        # Evaluate initial estimator on train/val/test for uncertainty_quantification task
        if cfg.task.task_name == "uncertainty_quantification":
            logging.info("Evaluating initial estimator on train/val/test splits...")
            for split in ["train", "val", "testset"]:
                initial_result, num_samples = evaluate_uq(cfg, generate_input, evaluate_func, intial_function_class, split)
                write_uq_metrics_row(cfg, initial_result, split, num_samples, iteration=-1)
                logging.info(f"Initial estimator {split} evaluation: AUC={initial_result.extra_metrics.get('AUC', 'N/A')}")

        # For saving
        os.makedirs(f"{cfg.logs_dir}/chats", exist_ok=True)
        os.makedirs(f"{cfg.logs_dir}/failed_chats", exist_ok=True)
        os.makedirs(f"{cfg.logs_dir}/programbank", exist_ok=True)
        os.makedirs(f"{cfg.logs_dir}/candidates", exist_ok=True)

        # Log the initial statistics from the programdatabase
        running_dict = get_pd_statistics(cfg, programdatabase, running_dict, -1)
    if cfg.wandb:
        wandb_log_running_dict(running_dict)

    if running_dict["flag_load_finetuned"] == 1:
        flag_load_finetuned = True
    else:
        flag_load_finetuned = False

    logging.info(f"Flag load finetuned: {flag_load_finetuned}")
    # Initialize model(s)
    if cfg.use_tgi:
        logging.info("====================== Inference Engine: TGI ======================")
        model_server_alive = False
    elif cfg.use_vllm:
        logging.info("====================== Inference Engine: vLLM ======================")
        model_server_alive = False
    elif cfg.use_api:
        logging.info("====================== Inference Engine: API ======================")
        model_server_alive = False
    else:
        model_initialized = False
        logging.info("====================== Inference Engine: Transformers ======================")

    logging.info(f"Starting with round number {starting_round}")
    for round_num in range(starting_round, cfg.num_rounds, cfg.num_cont_rounds):

        time_start = time.time()

        logging.info(f"STARTING ROUND NUM: {round_num}")
        logging.info(f"Prompt num: {running_dict['prompt_num']}")
        logging.info(f"Total functions generated so far: {running_dict['num_func_generated']}")
        logging.info(f"Total functions evaluated so far: {running_dict['num_func_evaluated']}")
        logging.info(
            f"Number of functions in the programbank: {programdatabase.total_num_programs}"
        )
        logging.info(f"Number of datatpoints in DPO data: {len(dpo_chats)}")
        logging.info(f"Best overall program score: {programdatabase.get_best_score}")
        logging.info(f"Best running_dict score: {running_dict['best_score']}")
        logging.info(
            f"Best running_dict best_true_score_from_score: {running_dict['best_true_score_from_score']}"
        )
        running_dict["round_num"] = copy.deepcopy(round_num)

        if cfg.use_tgi:
            if model_server_alive:
                logging.info(f"TGI Model server is still alive, continuing...")
            else:
                logging.info(f"Starting TGI model server...")
                server_pids, server_ports = initialize_models_server(cfg, flag_load_finetuned)
                model_server_alive = True
        elif cfg.use_vllm:
            if model_server_alive:
                logging.info(f"VLLM model server is still alive, continuing...")
            else:
                logging.info(f"Starting vLLM model server...")
                server_pids, server_ports = initialize_models_server(cfg, flag_load_finetuned, use_vllm=True)
                model_server_alive = True
        elif cfg.use_api:
            # API mode doesn't require server initialization
            if model_server_alive:
                logging.info(f"API mode initialized, continuing...")
            else:
                logging.info(f"Initializing API mode...")
                server_pids = []  # No server PIDs for API mode
                server_ports = [None]  # Dummy port list for compatibility
                model_server_alive = True
        else:
            if model_initialized:
                logging.info(f"Models already initialized, continuing...")
            else:
                model, tokenizer, sampling_params = initialize_models(cfg, flag_load_finetuned)
                logging.info("Model(s) initialized")
                model_initialized = True

        # GENERATE FUNCTIONS
        logging.info("-" * 10)
        logging.info("GENERATING CANDIDATES")
        (
            chats_batch,
            island_id_prompt_batch,
            prompt_scores_batch,
            temperatures_batch,
            probabilities_batch,
            prompt_nums_batch,
            running_dict,
        ) = generate_batch_prompts(cfg, programdatabase, running_dict, round_num)

        # Create a queue for functions, lists for failed and passed functions and a dict for running stats
        function_classes_to_eval = multiprocessing.Queue()
        # For storing errors from consumers
        error_queue = multiprocessing.Queue()
        manager = multiprocessing.Manager()
        failed_function_classes_global = manager.list()
        passed_function_classes_global = manager.list()

        multiprocess_running_dict = manager.dict()
        multiprocess_running_dict["llm_generation_time"] = manager.list()
        multiprocess_running_dict["num_func_generated"] = 0
        multiprocess_running_dict["num_func_evaluated"] = 0

        # Create a termination event, if set we know there was an error and the run should stop
        termination_event = multiprocessing.Event()

        # Start the consumer processes
        logging.info("-" * 10)
        logging.info("STARTING CONSUMERS")
        
        # For tasks with large tensor data (e.g., UQ), use lazy loading to avoid
        # serializing gigabytes of data over multiprocessing pipes. Subprocesses
        # will load data from disk instead, which is much faster.
        use_lazy_load = cfg.task.task_name == "uncertainty_quantification"
        if use_lazy_load:
            input_struct_or_spec = {"_lazy_load": True, "split": "train"}
            logging.info("Using lazy loading for input data (subprocess will load from disk)")
        else:
            input_struct_or_spec = input_struct
        
        consumers = []
        for worker_id in range(cfg.num_workers):
            p = multiprocessing.Process(
                target=Consumer,
                args=(
                    cfg,
                    evaluate_func,
                    function_classes_to_eval,
                    passed_function_classes_global,
                    failed_function_classes_global,
                    worker_id,
                    termination_event,
                    input_struct_or_spec,
                    round_num,
                    multiprocess_running_dict,
                ),
            )
            consumers.append(p)
            p.start()

        # Start the producer thread
        logging.info("-" * 10)
        logging.info("STARTING THE PRODUCER")
        if cfg.use_tgi or cfg.use_vllm or cfg.use_api:
            producer_thread = threading.Thread(
                target=Producer,
                args=(
                    cfg,
                    function_classes_to_eval,
                    failed_function_classes_global,
                    chats_batch,
                    prompt_scores_batch,
                    island_id_prompt_batch,
                    temperatures_batch,
                    probabilities_batch,
                    prompt_nums_batch,
                    multiprocess_running_dict,
                    termination_event,
                    flag_load_finetuned,
                    server_ports,
                ),
            )
        else:
            producer_thread = threading.Thread(
                target=Producer,
                args=(
                    cfg,
                    # server_ports,
                    function_classes_to_eval,
                    failed_function_classes_global,
                    chats_batch,
                    prompt_scores_batch,
                    island_id_prompt_batch,
                    temperatures_batch,
                    probabilities_batch,
                    prompt_nums_batch,
                    multiprocess_running_dict,
                    termination_event,
                    flag_load_finetuned,
                    model,
                    tokenizer,
                    sampling_params,
                ),
            )
        producer_thread.start()

        # Wait for the producer to finish
        producer_thread.join()

        # Wait for the consumers to finish and clean up
        consumers_finish_and_cleanup(
            cfg,
            consumers,
            function_classes_to_eval,
        )

        # Collect the results, will be registered at the end of the round
        passed_function_classes = copy.deepcopy(list(passed_function_classes_global))
        failed_function_classes = copy.deepcopy(list(failed_function_classes_global))

        logging.info("###################")
        logging.info(f"Num of passed functions: {len(passed_function_classes)}")
        logging.info(f"Num of failed functions: {len(failed_function_classes)}")
        logging.info("###################")

        # Log every evaluated candidate's metrics to CSV from the parent process
        if cfg.task.task_name == "hallu":
            logging.info(f"======== cfg.logs_dir={cfg.logs_dir} =======")
            csv_file = os.path.join(cfg.logs_dir, "all_metrics.csv")
            new_file = not os.path.exists(csv_file)
            with open(csv_file, "a", newline="") as _f:
                writer = csv.writer(_f)
                if new_file:
                    writer.writerow(["function_num", "prompt_num", "auc", "precision", "recall", "invalid_responses_percentage", "auc_lower_bound", "auc_upper_bound"])
                for function_class in passed_function_classes:
                    em = getattr(function_class, "extra_metrics", {}) or {}
                    lower, upper = parse_auc_ci(em.get("auc_ci", "(nan, nan)"))
                    writer.writerow([
                        function_class.function_num,
                        function_class.prompt_num,
                        em.get("auc", ""),
                        em.get("precision_at_0_5", ""),
                        em.get("recall_at_0_5", ""),
                        em.get("invalid_responses_percentage", ""),
                        lower,
                        upper,
                    ])

        # Log every evaluated candidate's metrics to CSV for uncertainty_quantification task
        if cfg.task.task_name == "uncertainty_quantification":
            logging.info(f"======== cfg.logs_dir={cfg.logs_dir} =======")
            csv_file = os.path.join(cfg.logs_dir, "all_metrics_train.csv")
            new_file = not os.path.exists(csv_file)
            with open(csv_file, "a", newline="") as _f:
                writer = csv.writer(_f)
                if new_file:
                    writer.writerow([
                        "round_num", "function_num", "prompt_num",
                        "auc", "precision_at_median", "recall_at_median",
                        "num_samples", "auc_lower_bound", "auc_upper_bound"
                    ])
                for function_class in passed_function_classes:
                    em = getattr(function_class, "extra_metrics", {}) or {}
                    lower, upper = parse_auc_ci(em.get("AUC 95% confidence interval", "(nan, nan)"))
                    writer.writerow([
                        round_num,
                        getattr(function_class, "function_num", ""),
                        getattr(function_class, "prompt_num", ""),
                        em.get("AUC", ""),
                        em.get("Precision at median score threshold", ""),
                        em.get("Recall at median score threshold", ""),
                        em.get("Number of samples", ""),
                        lower,
                        upper,
                    ])

        # Evaluate top-1 best prompt on the TEST set for the hallu task after each iteration
        if cfg.task.task_name == "hallu":
            # Select candidates with valid AUC and invalid_responses_percentage < 1
            candidates = [
                fc for fc in passed_function_classes
                if isinstance(getattr(fc, "extra_metrics", None), dict)
                and "auc" in fc.extra_metrics
                and fc.extra_metrics.get("invalid_responses_percentage", 100.0) < 1.0
            ]

            if len(candidates) > 0:
                # Highest ROC-AUC wins
                best_fc = max(candidates, key=lambda fc: fc.extra_metrics.get("auc", float("-inf")))
                prompt_text = best_fc.extra_metrics.get("prompt_text", "")

                test_csv_path = os.path.join(cfg.logs_dir, "test_metrics.csv")
                already_evaluated = False
                if os.path.exists(test_csv_path):
                    with open(test_csv_path, "r", newline="") as tf:
                        reader = csv.DictReader(tf)
                        for row in reader:
                            if row.get("prompt_text", "") == prompt_text:
                                already_evaluated = True
                                break

                if not already_evaluated:
                    test_result, num_test_samples = evaluate_hallu(cfg, generate_input, evaluate_func, best_fc, "testset")
                    for split in ["val", "testset"]:
                        test_result, num_test_samples = evaluate_hallu(cfg, generate_input, evaluate_func, best_fc, split)
                        write_test_metrics_row(cfg, test_result, split, num_test_samples, iteration=round_num)

        # Evaluate top-1 best estimator on val/test for uncertainty_quantification task
        if cfg.task.task_name == "uncertainty_quantification":
            # Select candidates with valid AUC
            candidates = [
                fc for fc in passed_function_classes
                if isinstance(getattr(fc, "extra_metrics", None), dict)
                and "AUC" in fc.extra_metrics
            ]

            if len(candidates) > 0:
                # Highest ROC-AUC wins
                best_fc = max(candidates, key=lambda fc: fc.extra_metrics.get("AUC", float("-inf")))
                best_auc = best_fc.extra_metrics.get("AUC", None)

                csv_path = os.path.join(cfg.logs_dir, "uq_split_metrics.csv")
                already_evaluated = False
                if os.path.exists(csv_path):
                    with open(csv_path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Check if same AUC already evaluated at this iteration
                            try:
                                if (int(row.get("iteration", -999)) == round_num and
                                    float(row.get("auc", -1)) == best_auc):
                                    already_evaluated = True
                                    break
                            except (ValueError, TypeError):
                                pass

                if not already_evaluated:
                    logging.info(f"Evaluating best estimator (train AUC={best_auc}) on val/test splits...")
                    split2result = {}
                    for split in ["train", "val", "testset"]:
                        result, num_samples = evaluate_uq(cfg, generate_input, evaluate_func, best_fc, split)
                        write_uq_metrics_row(cfg, result, split, num_samples, iteration=round_num)
                        split2result[split] = result
                        logging.info(f"UQ {split} evaluation: AUC={result.extra_metrics.get('AUC', 'N/A')}")
                    
                    with open(f"{cfg.logs_dir}/candidates/candidate_round_{round_num}_function_{getattr(best_fc, 'function_num', 0)}.txt", "w") as f:
                        f.write(f"# Iteration: {round_num}\n")
                        f.write(f"# Test AUC: {split2result['testset'].score}, n_samples: {split2result['testset'].extra_metrics.get('Number of samples', 'N/A')}\n")
                        f.write(f"# Test AUC 95% confidence interval (bootstrap): {split2result['testset'].extra_metrics.get('AUC 95% confidence interval (bootstrap)', 'N/A')}\n")
                        f.write(f"# Val AUC: {split2result['val'].score}, n_samples: {split2result['val'].extra_metrics.get('Number of samples', 'N/A')}\n")
                        f.write(f"# Val AUC 95% confidence interval (bootstrap): {split2result['val'].extra_metrics.get('AUC 95% confidence interval (bootstrap)', 'N/A')}\n")
                        f.write(f"# Train AUC: {split2result['train'].score}, n_samples: {split2result['train'].extra_metrics.get('Number of samples', 'N/A')}\n")
                        f.write(f"# Train AUC 95% confidence interval (bootstrap): {split2result['train'].extra_metrics.get('AUC 95% confidence interval (bootstrap)', 'N/A')}\n")
                        f.write(f"{split2result['testset'].imports_str}\n")
                        f.write(f"{split2result['testset'].function_str}\n")

        # Evaluate top-1 best estimator on val/test for uncertainty_quantification_multi task
        if cfg.task.task_name == "uncertainty_quantification_multi":
            # Select candidates with valid AUC
            candidates = [
                fc for fc in passed_function_classes
                if isinstance(getattr(fc, "extra_metrics", None), dict)
                and "datasets" in fc.extra_metrics
            ]

            if len(candidates) > 0:
                # Highest (mean_prr + min_prr) / 2 score wins
                best_fc = max(candidates, key=lambda fc: fc.score)
                best_score = best_fc.score

                csv_path = os.path.join(cfg.logs_dir, "uq_multi_split_metrics.csv")
                already_evaluated = False
                if os.path.exists(csv_path):
                    with open(csv_path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Check if same score already evaluated at this iteration
                            try:
                                if (int(row.get("iteration", -999)) == round_num and
                                    float(row.get("score", -1)) == best_score):
                                    already_evaluated = True
                                    break
                            except (ValueError, TypeError):
                                pass

                if not already_evaluated:
                    logging.info(f"Evaluating best estimator (train (mean_prr + min_prr) / 2 score={best_score}) on val/test splits...")
                    split2result = {}
                    for split in ["train", "val", "testset"]:
                        result = evaluate_uq_multi(cfg, generate_input, evaluate_func, best_fc, split)
                        write_uq_multi_metrics_row(cfg, result, split, iteration=round_num)
                        split2result[split] = result
                        logging.info(f"UQ {split} evaluation: (mean_prr + min_prr) / 2 score={result.score}")
                    
                    program_text = f"{split2result['testset'].imports_str}\n{split2result['testset'].function_str}\n"

                    with open(f"{cfg.logs_dir}/candidates/candidate_round_{round_num}_function_{getattr(best_fc, 'function_num', 0)}.txt", "w") as f:
                        f.write(f"# Iteration: {round_num}\n")
                        for split in ["testset", "val", "train"]:
                            f.write(f"# {split.title()} per-dataset PRR: ")
                            split_dataset_results = split2result[split].extra_metrics['datasets']
                            for dataset in split_dataset_results.keys():
                                f.write(f"{dataset}: {100*split_dataset_results[dataset]['prediction_rejection_ratio']:.1f}, ")
                            f.write(f"\n")
                        f.write(program_text)


        # Clean up
        passed_function_classes_global[:] = []
        failed_function_classes_global[:] = []

        multiprocess_running_dict = dict(multiprocess_running_dict)
        running_dict["num_func_generated"] += multiprocess_running_dict["num_func_generated"]
        running_dict["times/llm_generation_time"] = np.mean(
            multiprocess_running_dict["llm_generation_time"]
        )
        manager.shutdown()
        del passed_function_classes_global
        del failed_function_classes_global
        del multiprocess_running_dict
        del manager

        # if termination_event.is_set():
        #     raise RuntimeError("Termination event was set, there was an error.")

        logging.info(
            f"Finished with continuous sampling and evaluation for {cfg.num_cont_rounds} rounds"
        )

        # No finetuning in this project
        running_dict["training_flag_round"] = 0

        logging.info("-" * 10)
        logging.info("REGISTERING PROGRAMS")

        # Log the average score of passed and failed functions in this batch
        running_dict = calculate_avg_score(
            running_dict, passed_function_classes, failed_function_classes
        )
        # Log the average score of only passed functions in this batch
        running_dict = calculate_avg_passed_score(running_dict, passed_function_classes)

        # Register the sampled functions in the programdatabase
        programdatabase, running_dict, _ = register_programs(
            cfg, running_dict, passed_function_classes, failed_function_classes, programdatabase
        )

        # Get programdatabase stats
        running_dict = get_pd_statistics(cfg, programdatabase, running_dict, round_num)

        # Save to file
        save_pd_to_file(cfg, programdatabase, round_num)
        save_in_logsdir(
            cfg.logs_dir,
            running_dict,
            programdatabase,
            dpo_chats,
            round_num,
            input_struct,
        )
        logging.info(f"Saved to logs directory")

        time_end = time.time()

        logging.info(f"\n\nTime taken for one round: {int(time_end - time_start)} seconds \n\n")
        running_dict[f"times/{cfg.num_cont_rounds}rounds_time_taken"] = int(time_end - time_start)
        running_dict["num_func/num_func_generated"] = running_dict["num_func_generated"]
        running_dict["num_func/num_func_evaluated"] = running_dict["num_func_evaluated"]
        running_dict["num_func/num_func_failed"] = running_dict["num_func_failed"]
        running_dict["num_func/num_func_passed"] = running_dict["num_func_passed"]
        running_dict["num_func/num_func_correct"] = running_dict["num_func_correct"]

        running_dict["traindata/num_func_dpo"] = len(dpo_chats)
        if cfg.wandb:
            wandb_log_running_dict(running_dict)

        # Save a flag_resume that indicates to preempted jobs that it should resume from the last round
        with open(f"{cfg.logs_dir}/flag_resume.txt", "w") as f:
            pass

        del passed_function_classes
        del failed_function_classes
        del chats_batch
        del island_id_prompt_batch
        del prompt_scores_batch
        del temperatures_batch
        del probabilities_batch
        del prompt_nums_batch

        logging.info("\n\n")
        logging.info(f"ROUND {round_num} FINISHED")

    logging.info("FINISHED ALL ROUNDS")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

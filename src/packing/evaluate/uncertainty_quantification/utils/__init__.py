"""Utility modules for uncertainty quantification evaluation."""

from .bootstrap_utils import (
    delong_roc_variance,
    delong_auc_ci,
    bootstrap_auc_ci,
    generate_bootstrap_indices,
    bootstrap_prr_ci,
)
from .data_utils import (
    normalize_dataframe_classification,
    normalize_dataframe_generation,
    validate_csv_columns,
)
from .metric_selection import (
    detect_dataset_type_from_filename,
    get_metric_for_dataset_type,
    select_best_available_metric,
)
from .eval_utils import (
    load_model_and_tokenizer,
    apply_chat_template,
    pack_single_sample_raw_states,
)

__all__ = [
    # Bootstrap utilities
    "delong_roc_variance",
    "delong_auc_ci",
    "bootstrap_auc_ci",
    "generate_bootstrap_indices",
    "bootstrap_prr_ci",
    # Data utilities
    "normalize_dataframe_classification",
    "normalize_dataframe_generation",
    "validate_csv_columns",
    # Metric selection
    "detect_dataset_type_from_filename",
    "get_metric_for_dataset_type",
    "select_best_available_metric",
    # Evaluation utilities
    "load_model_and_tokenizer",
    "apply_chat_template",
    "pack_single_sample_raw_states",
]

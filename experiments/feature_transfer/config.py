import json
from pathlib import Path

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())
MODELS = {key: (value["id"], value["revision"]) for key, value in PROTOCOL["models"].items()}
DATASETS = {item["dataset_key"]: item for item in PROTOCOL["datasets"]}
DATASET_ORDER = tuple(DATASETS)
METHOD_NAMES = tuple(PROTOCOL["methods"])
ARCHIVE_SHA256 = PROTOCOL["archive_sha256"]
FIELDS = ("sequence_logprobs", "greedy_logprobs", "token_entropies", "top_k_logits")

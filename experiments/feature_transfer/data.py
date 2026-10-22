import csv
import hashlib
import io
from zipfile import ZipFile

from .config import ARCHIVE_SHA256, DATASETS
from .io import sha256


def normalize(claim):
    return claim.replace("\r\n", "\n").replace("\r", "\n").strip()[:300]


def load_datasets(archive_path, keys, limit=None):
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("The dataset archive differs from the recorded experiment.")
    datasets = {}
    with ZipFile(archive_path) as archive:
        for key in keys:
            pin = DATASETS[key]
            content = archive.read(pin["member"])
            if hashlib.sha256(content).hexdigest() != pin["sha256"]:
                raise ValueError(f"Source checksum mismatch: {key}")
            records = csv.DictReader(io.StringIO(content.decode("utf-8")))
            rows = [
                {"row_id": f"{key}:{index}", "row_index": index, "dataset_key": key,
                 "label": int(float(record["is_hallu"])), "claim": normalize(record["claim"])}
                for index, record in enumerate(records)
            ]
            valid = (len(rows) == pin["rows"]
                     and sum(row["label"] for row in rows) == pin["is_hallu_1"]
                     and all(row["label"] in (0, 1) and row["claim"] for row in rows))
            if not valid:
                raise ValueError(f"Source rows or labels differ from the experiment: {key}")
            datasets[key] = {"pin": pin, "rows": rows if limit is None else rows[:limit]}
    return datasets

"""Summarize saved GPU samples and persisted request usage without contacting a pod.

GPU utilization is an instantaneous nvidia-smi sample approximately once a minute,
not a continuous trace. Cost is an elapsed compute estimate, not a billing receipt.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def utc(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("Timestamp must include its UTC offset")
    return result.astimezone(dt.timezone.utc)


def json_lines(data, name, warnings):
    lines = data.splitlines()
    for index, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1 or data.endswith(b"\n"):
                raise
            warnings.append("Excluded incomplete final JSONL record: " + name)


def gpu_summary(records, as_of, threshold):
    devices, moments, identities = defaultdict(list), set(), set()
    for record in records:
        time = utc(record["utc"])
        if time > as_of:
            continue
        fields = record["fields"]
        for index, values in enumerate(csv.reader(io.StringIO(record["values"]))):
            if not values:
                continue
            if len(values) != len(fields) or (time, index) in identities:
                raise ValueError("Malformed or duplicate telemetry sample")
            values = dict(zip(fields, map(float, values)))
            if (not all(math.isfinite(x) for x in values.values())
                    or not 0 <= values["utilization_percent"] <= 100
                    or values["memory_MiB"] < 0):
                raise ValueError("Invalid GPU telemetry values")
            identities.add((time, index))
            devices[index].append((time, values))
            moments.add(time)
    rows = []
    for index, samples in sorted(devices.items()):
        samples.sort()
        utilization = [values["utilization_percent"] for _, values in samples]
        loaded = [value for value in utilization if value >= threshold]
        rows.append(dict(gpu_index=index, samples=len(samples), first_utc=samples[0][0].isoformat(),
            last_utc=samples[-1][0].isoformat(), mean_sampled_utilization_percent=statistics.mean(utilization),
            loaded_samples=len(loaded), loaded_sample_fraction=len(loaded)/len(samples),
            mean_utilization_loaded_samples_percent=statistics.mean(loaded) if loaded else None,
            peak_memory_MiB=max(values["memory_MiB"] for _, values in samples),
            peak_memory_GiB=max(values["memory_MiB"] for _, values in samples)/1024,
            mean_sampled_power_W=statistics.mean(values["power_W"] for _, values in samples),
            peak_temperature_C=max(values["temperature_C"] for _, values in samples)))
    moments = sorted(moments)
    intervals = [(later-earlier).total_seconds() for earlier, later in zip(moments, moments[1:])]
    return rows, dict(unique_sample_times=len(moments), median_interval_seconds=statistics.median(intervals) if intervals else None,
                      maximum_gap_seconds=max(intervals) if intervals else None)


def request_summary(records):
    totals = dict(persisted_requests=len(records), requests_with_token_usage=0, prompt_tokens=0,
                  completion_tokens=0, total_tokens=0, max_prompt_tokens=0, max_completion_tokens=0,
                  max_actual_prompt_plus_completion_tokens=0)
    latencies = []
    for row in records:
        usage = row.get("usage", {})
        values = [usage.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0) for value in values):
            raise ValueError("Malformed token usage in request ledger")
        prompt, completion, total = values
        if prompt is not None and completion is not None:
            totals["requests_with_token_usage"] += 1
            totals["prompt_tokens"] += prompt
            totals["completion_tokens"] += completion
            totals["total_tokens"] += total if total is not None else prompt+completion
            totals["max_prompt_tokens"] = max(totals["max_prompt_tokens"], prompt)
            totals["max_completion_tokens"] = max(totals["max_completion_tokens"], completion)
            totals["max_actual_prompt_plus_completion_tokens"] = max(totals["max_actual_prompt_plus_completion_tokens"], prompt+completion)
        seconds = row.get("generation_seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0:
            latencies.append(seconds)
    totals.update(requests_missing_token_usage=len(records)-totals["requests_with_token_usage"],
        median_request_seconds=statistics.median(latencies) if latencies else None,
        max_request_seconds=max(latencies) if latencies else None,
        sum_request_seconds=sum(latencies))
    return totals


def elapsed_cost(start, as_of, hourly):
    if hourly < 0 or not math.isfinite(hourly) or as_of < start:
        raise ValueError("Invalid compute-cost interval or rate")
    seconds = (as_of-start).total_seconds()
    return dict(start_utc=start.isoformat(), as_of_utc=as_of.isoformat(), elapsed_seconds=seconds,
                hourly_cost_usd=hourly, estimated_elapsed_compute_usd=seconds/3600*hourly)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/overnight20261021/analysis_manifest.json")
    parser.add_argument("--telemetry", type=Path, default=ROOT / "out/overnight20261021/gpu_telemetry.jsonl")
    parser.add_argument("--budget", type=Path, default=ROOT / "results/overnight20261021/resource_budget.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/overnight20261021/resource_summary")
    parser.add_argument("--as-of-utc", help="Cost endpoint; default is the latest saved telemetry sample, not current time")
    parser.add_argument("--load-threshold-percent", type=float, default=10)
    args = parser.parse_args()
    if not 0 <= args.load_threshold_percent <= 100:
        parser.error("load threshold must be between 0 and 100")
    hashes, warnings = {}, []

    def read(path):
        data = path.read_bytes()
        hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = hashlib.sha256(data).hexdigest()
        return data

    read(Path(__file__))
    manifest, budget = json.loads(read(args.manifest)), json.loads(read(args.budget))
    telemetry = list(json_lines(read(args.telemetry), str(args.telemetry), warnings))
    if not telemetry and not args.as_of_utc:
        raise ValueError("No saved telemetry; provide --as-of-utc for the elapsed-cost interval")
    as_of = utc(args.as_of_utc) if args.as_of_utc else max(utc(row["utc"]) for row in telemetry)
    gpus, cadence = gpu_summary(telemetry, as_of, args.load_threshold_percent)
    jobs, all_records, seen_ids, seen_paths = [], [], set(), set()
    for job in manifest["jobs"]:
        folder = ROOT / job["expected_logs"]
        if job["id"] in seen_ids or folder in seen_paths:
            raise ValueError("Duplicate job identity or ledger path would double-count usage")
        seen_ids.add(job["id"])
        seen_paths.add(folder)
        ledger = folder / "candidates.jsonl"
        records = list(json_lines(read(ledger), str(ledger), warnings)) if ledger.exists() else []
        identities = [(row["round"], row["index"]) for row in records]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate request identity in " + str(ledger))
        jobs.append(dict(job=job["id"], phase=job.get("analysis_phase", job["phase"]), **request_summary(records)))
        all_records.extend(records)
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), model_calls=False,
        telemetry_interpretation="Instantaneous nvidia-smi samples approximately once a minute; means and loaded fractions describe samples, not continuous GPU busy time. Allocated memory includes reserved KV cache.",
        load_threshold_percent=args.load_threshold_percent,
        cost=elapsed_cost(utc(budget["start_utc"]), as_of, float(budget["hourly_cost_usd"])),
        cost_interpretation="Elapsed compute estimate at the recorded total pod rate; excludes storage/network and does not certify actual billing or pod stop time.",
        token_interpretation="Counts each persisted candidates.jsonl request once across analysis-manifest jobs. Responses still awaiting evaluation/persistence and unlisted jobs are excluded. Request time sums include overlapping requests and are not wall time.",
        telemetry_cadence=cadence, gpus=gpus, requests=request_summary(all_records), jobs=jobs,
        warnings=warnings, input_sha256=hashes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resource_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (args.output / "resource_jobs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]) if jobs else ["job"])
        writer.writeheader()
        writer.writerows(jobs)
    cost, requests = report["cost"], report["requests"]
    lines = ["# Overnight resource usage", "",
        f"Saved telemetry through {as_of.isoformat()}. Elapsed compute estimate: **${cost['estimated_elapsed_compute_usd']:.2f}** over {cost['elapsed_seconds']/3600:.2f} hours at ${cost['hourly_cost_usd']:.2f}/hour.", "",
        "| GPU | Samples | Mean sampled utilization | Samples ≥ load threshold | Mean utilization in loaded samples | Peak allocated memory |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for gpu in gpus:
        loaded_mean = gpu["mean_utilization_loaded_samples_percent"]
        lines.append(f"| {gpu['gpu_index']} | {gpu['samples']} | {gpu['mean_sampled_utilization_percent']:.1f}% | {100*gpu['loaded_sample_fraction']:.1f}% | {f'{loaded_mean:.1f}%' if loaded_mean is not None else 'undefined'} | {gpu['peak_memory_GiB']:.2f} GiB |")
    lines += ["", f"Load threshold: {args.load_threshold_percent:g}%. " + report["telemetry_interpretation"], "",
        f"Persisted requests: **{requests['persisted_requests']}**; prompt tokens **{requests['prompt_tokens']:,}**; completion tokens **{requests['completion_tokens']:,}**; total **{requests['total_tokens']:,}**. Requests without reported usage: {requests['requests_missing_token_usage']}.", "",
        report["token_interpretation"], "", report["cost_interpretation"]]
    (args.output / "RESOURCE_USAGE.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(args.output), gpus=len(gpus), persisted_requests=requests["persisted_requests"],
                         estimated_elapsed_compute_usd=cost["estimated_elapsed_compute_usd"], model_calls=False)))


if __name__ == "__main__":
    main()

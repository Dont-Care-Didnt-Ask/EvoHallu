"""Move the same three fixed-feedback jobs earlier; no additional requests.

Default is a read-only plan. --prepare freezes the exact source subset.
--execute reserves original log directories and runs the unchanged controller.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path

from scripts import overnight_followups as common
from scripts import run_overnight_parallel as controller

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"results/next_experiments/run_manifest.json"
MANIFEST=ROOT/"results/next_experiments/feedback_overlap_manifest.json"
STATE=ROOT/"results/next_experiments/feedback_overlap_state.json"
RECEIPT=ROOT/"results/overnight20261021/feedback_overlap_receipt.json"
MARKER=".feedback_overlap_owner.json"


def job_sha(job):
    return hashlib.sha256(json.dumps(job,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def prepare_manifest(source_path=SOURCE,manifest_path=MANIFEST,root=ROOT):
    source=common.read(source_path)
    if manifest_path.exists():
        value=common.read(manifest_path)
        common.validate_feedback_overlap_manifest(value,source,common.sha(source_path))
        return value
    value=copy.deepcopy(source)
    value.pop("later_design_only",None)
    value.update(created_utc=common.now().isoformat(),status="prepared_only_same_original_feedback_jobs",
                 study="resource_only_earlier_schedule_of_existing_feedback_controls",
                 jobs=[job for job in source["jobs"] if job.get("phase")=="feedback_control"],
                 requests_by_phase={"feedback_control":180},parent_source_manifest=str(source_path.relative_to(root)),
                 parent_source_manifest_sha256=common.sha(source_path),additional_requests=0,
                 analysis_identity_rule="Keep original source job identities; this is scheduling metadata only",
                 ownership_rule="Reserve all original output directories; original main controller refuses still-partial reserved jobs")
    common.validate_feedback_overlap_manifest(value,source,common.sha(source_path))
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    with manifest_path.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    return value


def verify_early_source_window(source,source_state,root=ROOT):
    if source_state.get("manifest_sha256")!=common.sha(root/"results/next_experiments/run_manifest.json"):
        raise ValueError("Running main controller has a different source manifest")
    if common.terminal_evidence(source_state) or not common.process_alive(source_state.get("controller_pid",0)):
        raise ValueError("Main source controller must still be active")
    if source_state.get("ports")!=[18000,18001] or len(source_state.get("running",[]))>2:
        raise ValueError("Main source must use the expected two replicas and at most two search jobs")
    if any(event.get("event")=="batch_started" and event.get("batch")=="feedback_control" for event in source_state.get("events",[])):
        raise ValueError("Main source controller has already reached feedback; no overlapping launch allowed")
    future=0
    for job in source["jobs"]:
        folder=root/job["expected_logs"]
        if job.get("phase")=="feedback_control":
            if folder.exists() and any(folder.iterdir()):
                raise ValueError("Feedback already has artifacts; refusing ownership takeover")
        elif job.get("mode")=="evolution" and not (folder.exists() and any(folder.iterdir())):
            future+=1
    if future<2:
        raise ValueError("Require at least two unstarted adaptive jobs before reserving feedback")
    return dict(unstarted_adaptive_jobs=future,source_controller_pid=source_state["controller_pid"],
                source_feedback_phase_not_started=True)


def verify_prior_overlap_owners_terminal(root=ROOT):
    proof={}
    for stem,manifest_name in [("sampler_execution_state","sampler_manifest"),("interface_pilot_state","interface_pilot_manifest")]:
        path=root/"results/next_experiments"/(stem+".json")
        if not path.exists():continue
        manifest_path=path.with_name(manifest_name+".json")
        state=common.read(path)
        if not manifest_path.exists() or state.get("manifest_sha256")!=common.sha(manifest_path):
            raise ValueError("Prior overlapping controller manifest hash mismatch")
        evidence=common.terminal_evidence(state)
        if evidence is None:
            raise RuntimeError("Prior overlap still owns writer capacity: "+stem)
        proof[stem]=dict(manifest_sha256=state["manifest_sha256"],terminal_evidence=evidence)
    return proof


@contextlib.contextmanager
def owned_feedback_directories(manifest,manifest_sha256,root=ROOT):
    """Prevent the original main controller from starting any queued duplicate.

    The trial runner only refuses existing candidate ledgers, so these metadata
    markers do not alter its config/prompts. The overlap controller alone treats
    a marker-only directory as pending. All ordinary partial protections remain.
    """
    markers=[]
    original=controller.inspect_job
    jobs={job["id"]:job for job in manifest["jobs"]}
    payloads={}
    try:
        for job in manifest["jobs"]:
            folder=root/job["expected_logs"]
            if folder.exists() and any(folder.iterdir()):
                raise ValueError("Refusing to reserve an existing feedback directory")
            folder.mkdir(parents=True,exist_ok=True)
            marker=folder/MARKER
            payload=dict(job_id=job["id"],job_sha256=job_sha(job),manifest_sha256=manifest_sha256,
                         parent_source_manifest_sha256=manifest["parent_source_manifest_sha256"],owner_pid=os.getpid())
            with marker.open("x") as stream:json.dump(payload,stream,sort_keys=True)
            markers.append(marker);payloads[job["id"]]=payload

        def inspect_owned(job,root=None):
            actual_root=Path(root) if root is not None else Path(owned_root)
            folder=actual_root/job["expected_logs"]
            marker=folder/MARKER
            if job.get("id") in jobs and job==jobs[job["id"]] and marker.exists():
                if common.read(marker)!=payloads[job["id"]]:
                    raise ValueError("Feedback reservation ownership changed")
                if set(folder.iterdir())=={marker}:
                    return "pending"
            return original(job,actual_root)

        owned_root=root
        controller.inspect_job=inspect_owned
        yield markers
    finally:
        controller.inspect_job=original
        # Failed-before-start reservations must not obstruct the main controller.
        # Real partial/completed artifacts retain their ownership metadata.
        for marker in markers:
            if marker.exists() and set(marker.parent.iterdir())=={marker}:
                marker.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare",action="store_true")
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()
    os.chdir(ROOT)
    if args.prepare:prepare_manifest()
    if not MANIFEST.exists():
        if args.execute:raise FileNotFoundError("Prepare the immutable feedback subset before execution")
        print(json.dumps(dict(status="dry_plan",jobs=3,additional_requests=0,requests_moved_earlier=180)))
        return 0
    source,manifest=common.read(SOURCE),common.read(MANIFEST)
    common.validate_feedback_overlap_manifest(manifest,source,common.sha(SOURCE))
    if not args.execute:
        print(json.dumps(dict(status="prepared_not_launched",manifest=str(MANIFEST),manifest_sha256=common.sha(MANIFEST),jobs=3,
            additional_requests=0,source_configs_modified=False,command="PYTHONPATH=src:. .venv/bin/python scripts/run_feedback_overlap.py --execute")))
        return 0
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    with RECEIPT.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists():
            raise FileExistsError("An overlap execution state already exists; inspect it instead of relaunching")
        early=verify_early_source_window(source,common.read(common.SOURCE_STATE))
        prior_owners=verify_prior_overlap_owners_terminal()
        # Validate all pinned core/config/data bytes before reserving directories.
        controller.verify_inputs(manifest,MANIFEST,manifest["jobs"],ROOT)
        controller.verify_servers([18000,18001],manifest["writer"])
        state_args=argparse.Namespace(state=STATE,deadline_utc=common.DEADLINE,reserve_minutes=30,
            max_job_minutes=90,min_job_minutes=30,ports=[18000,18001])
        receipt=dict(status="reserved",started_utc=common.now().isoformat(),pid=os.getpid(),
            manifest=str(MANIFEST.relative_to(ROOT)),manifest_sha256=common.sha(MANIFEST),
            parent_source_manifest_sha256=common.sha(SOURCE),same_original_job_ids=[j["id"] for j in manifest["jobs"]],
            extra_requests=0,source_preflight=early,analysis_job_identities_unchanged=True,
            prior_overlap_owners_terminal=prior_owners,maximum_search_jobs=4,
            intervention="Scheduling only; marker-aware inspect hook confined to this controller process",
            helper_sha256=common.sha(Path(__file__)),controller_sha256=common.sha(ROOT/"scripts/run_overnight_parallel.py"))
        common.atomic(RECEIPT,receipt)
        try:
            with owned_feedback_directories(manifest,common.sha(MANIFEST)):
                receipt["status"]="running";common.atomic(RECEIPT,receipt)
                code=controller.execute(manifest,MANIFEST,controller.ordered_batches(manifest),state_args)
            receipt.update(status="finished" if code==0 else "partial_or_failed",exit_code=code)
            return code
        except BaseException as exc:
            receipt.update(status="failed",error=repr(exc))
            raise
        finally:
            receipt["finished_utc"]=common.now().isoformat()
            receipt["full_original_feedback_jobs"]=[j["id"] for j in manifest["jobs"] if common.full_budget(j)]
            common.atomic(RECEIPT,receipt)


if __name__=="__main__":
    raise SystemExit(main())

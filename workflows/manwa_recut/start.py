"""Build the chapter manifest and start the recut Distributed Map execution.

  python start.py --stack-name drawtoon-manwa-recut
                  [--input-prefix datasets/pages/single | datasets/pages/filtered]
                  [--output-prefix datasets/pages/single_recut | datasets/pages/filtered_cut]
                  [--limit 100] [--max-concurrency 3000]

The manifest is written to s3://drawtoon/datasets/_jobs/manwa_recut/<run_id>/manifest.jsonl
and the execution input points the state machine at that key.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_INPUT_PREFIX = "datasets/pages/single"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/single_recut"
SUFFIXES = ("_manwa", "_manhwa", "_manha", "_manhua")
JOB_PREFIX = "datasets/_jobs/manwa_recut"
AUDIT_PREFIX = "datasets/_stepfunctions_audit/manwa_recut"


def boto3_session(profile: str) -> boto3.Session:
    if profile:
        return boto3.Session(profile_name=profile)
    return boto3.Session()


def cfn_outputs(session: boto3.Session, stack_name: str) -> dict[str, str]:
    cfn = session.client("cloudformation")
    resp = cfn.describe_stacks(StackName=stack_name)
    out = {}
    for o in resp["Stacks"][0].get("Outputs") or []:
        out[o["OutputKey"]] = o["OutputValue"]
    return out


def list_series(s3, bucket: str, input_prefix: str) -> list[str]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{input_prefix}/", Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes") or []:
            name = (cp.get("Prefix") or "").rstrip("/").split("/")[-1]
            if name and any(name.endswith(s) for s in SUFFIXES):
                out.append(name)
    return sorted(out)


def list_chapters(s3, bucket: str, input_prefix: str, series: str) -> list[str]:
    """Return chapter keys for a series. Handles both layouts:
      (a) chapter-subdir layout: pages under <series>/<chapter>/page-NNNN.jpg
      (b) flat layout: pages under <series>/<episode-id>__page-NNNN.jpg
    For (b) we extract the episode-id prefix (everything before "__page-")
    and use that as the "chapter key" — the handler then loads pages by
    matching the same prefix.
    """
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{input_prefix}/{series}/", Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes") or []:
            out.append((cp.get("Prefix") or "").rstrip("/").split("/")[-1])
    if out:
        return sorted(out)
    prefixes: set[str] = set()
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{input_prefix}/{series}/"
    ):
        for o in page.get("Contents") or []:
            k = o["Key"]
            name = k.rsplit("/", 1)[-1]
            if "__page-" in name:
                prefixes.add(name.split("__page-", 1)[0])
    return sorted(prefixes)


def has_provenance(s3, bucket: str, output_prefix: str, series: str, chapter: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=f"{output_prefix}/{series}/{chapter}/_provenance.json")
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-name", default="drawtoon-manwa-recut")
    ap.add_argument("--profile", default="")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX)
    ap.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-concurrency", type=int, default=3000)
    ap.add_argument("--max-items-per-batch", type=int, default=1, help="chapters per Lambda invocation")
    ap.add_argument("--tolerated-failure-count", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="skip chapters with existing _provenance.json")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    session = boto3_session(args.profile.strip())
    s3 = session.client("s3")
    arn = cfn_outputs(session, args.stack_name)["RecutManwaStateMachineArn"]

    in_pre = args.input_prefix.strip().strip("/")
    out_pre = args.output_prefix.strip().strip("/")
    print(f"listing series + chapters under s3://{args.bucket}/{in_pre}/ ...", flush=True)
    t0 = time.perf_counter()
    series = list_series(s3, args.bucket, in_pre)
    with ThreadPoolExecutor(max_workers=32) as ex:
        per_series = list(ex.map(lambda s: (s, list_chapters(s3, args.bucket, in_pre, s)), series))
    jobs: list[tuple[str, str]] = [(s, c) for s, chs in per_series for c in chs]
    print(f"  series={len(series)}  chapters={len(jobs)}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    if args.resume:
        print(f"  checking which chapters already have output at {out_pre}/...", flush=True)
        def _has(job): return job, has_provenance(s3, args.bucket, out_pre, *job)
        skip = 0
        new_jobs = []
        with ThreadPoolExecutor(max_workers=32) as ex:
            for job, ok in ex.map(_has, jobs):
                if ok:
                    skip += 1
                else:
                    new_jobs.append(job)
        jobs = new_jobs
        print(f"  resume: skip={skip}  to-do={len(jobs)}", flush=True)

    if args.limit:
        jobs = jobs[: args.limit]
        print(f"  limit applied: {len(jobs)} chapters", flush=True)

    if not jobs:
        print("nothing to do.")
        return 0

    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest_key = f"{JOB_PREFIX}/{run_id}/manifest.jsonl"
    manifest_body = "\n".join(
        json.dumps({"series": s, "chapter": c}) for s, c in jobs
    ).encode("utf-8")
    s3.put_object(Bucket=args.bucket, Key=manifest_key, Body=manifest_body, ContentType="application/json")
    print(f"manifest: s3://{args.bucket}/{manifest_key}  ({len(jobs)} rows, {len(manifest_body):,} bytes)", flush=True)

    audit_prefix = f"{AUDIT_PREFIX}/{run_id}/"
    payload = {
        "source": {"bucket": args.bucket, "manifest_key": manifest_key},
        "input_prefix": in_pre,
        "output_prefix": out_pre,
        "jpeg_quality": int(args.jpeg_quality),
        "batch": {
            "max_concurrency": int(args.max_concurrency),
            "max_items_per_batch": int(args.max_items_per_batch),
        },
        "failure": {"tolerated_failure_count": int(args.tolerated_failure_count)},
        "audit": {"bucket": args.bucket, "prefix": audit_prefix},
    }
    print("execution input:")
    print(json.dumps(payload, indent=2))

    if args.dry_run:
        return 0

    sfn = session.client("stepfunctions")
    execution_name = f"manwa-recut-{run_id}"
    resp = sfn.start_execution(
        stateMachineArn=arn,
        name=execution_name,
        input=json.dumps(payload),
    )
    print(f"\nstarted: {resp['executionArn']}")
    print(f"        audit logs → s3://{args.bucket}/{audit_prefix}")


if __name__ == "__main__":
    sys.exit(main() or 0)

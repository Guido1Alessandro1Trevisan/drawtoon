"""Build the rejected-pages manifest, fire the probe Distributed Map, wait, and
report a height histogram.

Rejected set = keys present under {input_prefix}/<series>/... but NOT present
under {filtered_prefix}/<series>/... (set difference).

Usage:
  python start.py --stack-name drawtoon-probe-dims
                  [--limit 0]
                  [--max-concurrency 3000]
                  [--no-wait]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

try:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
except Exception:
    pass

DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
INPUT_PREFIX = "datasets/pages/single"
FILTERED_PREFIX = "datasets/pages/filtered"
SUFFIXES = ("_manwa", "_manhwa", "_manha", "_manhua")
JOB_PREFIX = "datasets/_jobs/probe_dims"
AUDIT_PREFIX = "datasets/_stepfunctions_audit/probe_dims"


def cfn_outputs(session: boto3.Session, stack_name: str) -> dict[str, str]:
    cfn = session.client("cloudformation")
    resp = cfn.describe_stacks(StackName=stack_name)
    out = {}
    for o in resp["Stacks"][0].get("Outputs") or []:
        out[o["OutputKey"]] = o["OutputValue"]
    return out


def list_series(s3, bucket: str, prefix: str) -> list[str]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{prefix}/", Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes") or []:
            name = (cp.get("Prefix") or "").rstrip("/").split("/")[-1]
            if name and any(name.endswith(s) for s in SUFFIXES):
                out.append(name)
    return sorted(out)


def list_all_keys(s3, bucket: str, prefix: str) -> list[str]:
    """List every object key under prefix/ (no delimiter)."""
    out: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents") or []:
            k = o.get("Key")
            if k:
                out.append(k)
    return out


def collect_series_keys(s3, bucket: str, series: str) -> tuple[str, set[str], set[str]]:
    """For one series, return (series, single_relpaths, filtered_relpaths).

    "relpath" = key after the {single|filtered}/<series>/ prefix.
    """
    sp = f"{INPUT_PREFIX}/{series}/"
    fp = f"{FILTERED_PREFIX}/{series}/"
    single_keys = list_all_keys(s3, bucket, sp)
    filtered_keys = list_all_keys(s3, bucket, fp)
    single_rel = {k[len(sp):] for k in single_keys if k.endswith(".jpg") or k.endswith(".jpeg") or k.endswith(".png") or k.endswith(".webp")}
    filtered_rel = {k[len(fp):] for k in filtered_keys if k.endswith(".jpg") or k.endswith(".jpeg") or k.endswith(".png") or k.endswith(".webp")}
    return series, single_rel, filtered_rel


def build_rejected_manifest(s3, bucket: str) -> list[str]:
    """Return list of full S3 keys for rejected manwa pages.

    Rejected = single keys whose relpath is NOT present in filtered/.
    """
    series_list = list_series(s3, bucket, INPUT_PREFIX)
    print(f"manwa-family series under {INPUT_PREFIX}/: {len(series_list)}", flush=True)
    rejected: list[str] = []
    series_done = 0
    total_single = 0
    total_filtered = 0
    with ThreadPoolExecutor(max_workers=64) as ex:
        futures = [ex.submit(collect_series_keys, s3, bucket, s) for s in series_list]
        for fut in as_completed(futures):
            series, single_rel, filtered_rel = fut.result()
            series_done += 1
            total_single += len(single_rel)
            total_filtered += len(filtered_rel)
            missing = single_rel - filtered_rel
            for rel in missing:
                rejected.append(f"{INPUT_PREFIX}/{series}/{rel}")
            if series_done % 25 == 0 or series_done == len(series_list):
                print(
                    f"  [{series_done}/{len(series_list)}] cumulative: single={total_single:,} filtered={total_filtered:,} rejected={len(rejected):,}",
                    flush=True,
                )
    return rejected


def wait_for_execution(sfn, arn: str) -> dict:
    """Poll until execution finishes; return final describe_execution payload."""
    last_status = None
    t0 = time.perf_counter()
    while True:
        resp = sfn.describe_execution(executionArn=arn)
        status = resp["status"]
        if status != last_status:
            print(f"  [{time.perf_counter() - t0:6.1f}s] status={status}", flush=True)
            last_status = status
        if status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
            return resp
        time.sleep(5)


def list_manifest_files(s3, bucket: str, prefix: str) -> list[str]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents") or []:
            out.append(o["Key"])
    return out


def read_result_writer_output(s3, bucket: str, audit_prefix: str) -> list[dict]:
    """ResultWriter v2 (with WriterConfig OutputType=JSONL) writes JSONL files
    under <audit_prefix><exec-name>/SUCCEEDED_*.jsonl plus a manifest. We read
    every SUCCEEDED_* file.
    """
    keys = list_manifest_files(s3, bucket, audit_prefix)
    succeeded_keys = [k for k in keys if "/SUCCEEDED_" in k and k.endswith(".jsonl")]
    print(f"  result writer files: {len(keys)} total, {len(succeeded_keys)} SUCCEEDED_*.jsonl", flush=True)
    rows: list[dict] = []
    for k in succeeded_keys:
        body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def histogram(heights: list[int], bin_size: int = 250) -> list[tuple[int, int, int]]:
    """Return [(bin_lo, bin_hi, count), ...] sorted ascending."""
    if not heights:
        return []
    counts: Counter[int] = Counter()
    for h in heights:
        b = (h // bin_size) * bin_size
        counts[b] += 1
    out = []
    for lo in sorted(counts):
        out.append((lo, lo + bin_size, counts[lo]))
    return out


def percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return int(round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-name", default="drawtoon-probe-dims")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--max-concurrency", type=int, default=3000)
    ap.add_argument("--tolerated-failure-count", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--existing-execution-arn", default="", help="skip manifest build, just wait on this arn")
    ap.add_argument("--existing-audit-prefix", default="", help="skip everything, just read results from this s3 prefix")
    ap.add_argument("--manifest-key", default="", help="pre-existing manifest s3 key (skip rebuild)")
    args = ap.parse_args()

    session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")
    sfn = session.client("stepfunctions")

    t_total0 = time.perf_counter()

    if args.existing_audit_prefix:
        rows = read_result_writer_output(s3, args.bucket, args.existing_audit_prefix.strip("/") + "/")
        report(rows, lambda_wall_time=0.0, total_wall_time=time.perf_counter() - t_total0)
        return 0

    arn = cfn_outputs(session, args.stack_name)["ProbeImageDimsStateMachineArn"]

    if args.existing_execution_arn:
        # Skip manifest, just wait & read
        exec_arn = args.existing_execution_arn
        run_id = exec_arn.split(":")[-1].replace("probe-dims-", "")
        audit_prefix = f"{AUDIT_PREFIX}/{run_id}/"
    else:
        # 1) build manifest
        t0 = time.perf_counter()
        if args.manifest_key:
            manifest_key = args.manifest_key
            # peek line count
            body = s3.get_object(Bucket=args.bucket, Key=manifest_key)["Body"].read()
            n_lines = sum(1 for ln in body.splitlines() if ln.strip())
            print(f"using existing manifest: s3://{args.bucket}/{manifest_key}  ({n_lines:,} rows)", flush=True)
            run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        else:
            rejected = build_rejected_manifest(s3, args.bucket)
            print(f"rejected total: {len(rejected):,}  ({time.perf_counter() - t0:.1f}s)", flush=True)
            if args.limit:
                rejected = rejected[: args.limit]
                print(f"  limit applied: {len(rejected):,}", flush=True)
            if not rejected:
                print("nothing to do.")
                return 0
            run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            manifest_key = f"{JOB_PREFIX}/{run_id}/manifest.jsonl"
            manifest_body = "\n".join(json.dumps({"key": k}) for k in rejected).encode("utf-8")
            s3.put_object(
                Bucket=args.bucket, Key=manifest_key, Body=manifest_body, ContentType="application/json"
            )
            print(
                f"manifest: s3://{args.bucket}/{manifest_key}  ({len(rejected):,} rows, {len(manifest_body):,} bytes)",
                flush=True,
            )

        audit_prefix = f"{AUDIT_PREFIX}/{run_id}/"
        payload = {
            "source": {"bucket": args.bucket, "manifest_key": manifest_key},
            "batch": {"max_concurrency": int(args.max_concurrency)},
            "failure": {"tolerated_failure_count": int(args.tolerated_failure_count)},
            "audit": {"bucket": args.bucket, "prefix": audit_prefix},
        }
        print("execution input:")
        print(json.dumps(payload, indent=2))

        # 2) start execution
        execution_name = f"probe-dims-{run_id}"
        t_exec0 = time.perf_counter()
        resp = sfn.start_execution(
            stateMachineArn=arn, name=execution_name, input=json.dumps(payload)
        )
        exec_arn = resp["executionArn"]
        print(f"started: {exec_arn}", flush=True)
        print(f"  audit logs -> s3://{args.bucket}/{audit_prefix}", flush=True)

        if args.no_wait:
            print("\n--no-wait set; exiting before completion.")
            print(f"resume with: --existing-execution-arn {exec_arn} --existing-audit-prefix {audit_prefix}")
            return 0

    # 3) wait
    print("\nwaiting for execution to complete...", flush=True)
    t_exec0 = time.perf_counter()
    final = wait_for_execution(sfn, exec_arn)
    lambda_wall_time = time.perf_counter() - t_exec0
    if final["status"] != "SUCCEEDED":
        # try to read whatever results we have anyway
        print(f"execution finished with status: {final['status']}", flush=True)
        print(json.dumps({k: str(v) for k, v in final.items()}, indent=2)[:2000])

    # 4) read & report
    print("\nreading result writer output...", flush=True)
    rows = read_result_writer_output(s3, args.bucket, audit_prefix)

    total_wall_time = time.perf_counter() - t_total0
    report(rows, lambda_wall_time=lambda_wall_time, total_wall_time=total_wall_time)
    return 0


def report(rows: list[dict], lambda_wall_time: float, total_wall_time: float) -> None:
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)

    ok = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("height"), int)]
    err = [r for r in rows if r.get("status") != "ok"]

    print(f"total rows           : {len(rows):,}")
    print(f"  successfully probed: {len(ok):,}")
    print(f"  errors             : {len(err):,}")

    print(f"\ntotal wall time      : {total_wall_time:.1f}s  ({total_wall_time/60:.2f} min)")
    print(f"lambda wall time     : {lambda_wall_time:.1f}s  ({lambda_wall_time/60:.2f} min)")
    if lambda_wall_time > 0 and len(ok) > 0:
        print(f"throughput           : {len(ok)/lambda_wall_time:.1f} images/sec")

    if not ok:
        print("\nno heights to histogram.")
        return

    heights = sorted(r["height"] for r in ok)
    print("\nheight stats (px):")
    print(f"  min     = {heights[0]}")
    print(f"  p10     = {percentile(heights, 10)}")
    print(f"  p25     = {percentile(heights, 25)}")
    print(f"  median  = {percentile(heights, 50)}")
    print(f"  p75     = {percentile(heights, 75)}")
    print(f"  p90     = {percentile(heights, 90)}")
    print(f"  p95     = {percentile(heights, 95)}")
    print(f"  p99     = {percentile(heights, 99)}")
    print(f"  max     = {heights[-1]}")
    print(f"  mean    = {statistics.fmean(heights):.1f}")
    print(f"  stdev   = {statistics.pstdev(heights):.1f}")

    print("\nheight histogram (250-px bins):")
    bins = histogram(heights, bin_size=250)
    max_count = max((c for _, _, c in bins), default=1)
    bar_w = 40
    for lo, hi, c in bins:
        bar = "#" * max(1, int(round(bar_w * c / max_count)))
        pct = 100.0 * c / len(heights)
        print(f"  {lo:>5}-{hi:<5} {c:>6}  ({pct:5.2f}%)  {bar}")

    if err:
        print("\nerror sample (up to 5):")
        for r in err[:5]:
            print(f"  {r.get('key', '?')}: {r.get('error', '?')}")


if __name__ == "__main__":
    sys.exit(main() or 0)

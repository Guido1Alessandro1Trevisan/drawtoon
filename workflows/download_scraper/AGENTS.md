# download_scraper Guide

## Scope

This guide applies to `workflows/download_scraper/`.

## Purpose

This workflow is a reusable, manifest-first downloader for authorized image
ingestion. The local machine may create manifests, deploy AWS resources, and
monitor runs. It should not be the long-running data-path downloader.

## Rules

- Source adapters produce JSONL manifests. AWS Lambda workers consume those
  manifests and write directly to S3.
- Do not hardcode credentials, cookies, proxy passwords, or session tokens.
  Use environment variables or AWS Secrets Manager.
- Do not guess protected CDN paths or forge signed URLs. Manifest rows must
  come from normal public or authorized browser/source flows.
- Proxies may be used only for source fetch routing. S3 uploads must go direct
  from AWS to S3.
- Keep run IDs, manifest paths, execution ARNs, and result counts documented in
  README or a handoff note after operational runs.


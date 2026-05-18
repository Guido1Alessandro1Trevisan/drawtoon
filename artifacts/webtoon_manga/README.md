# webtoon_manga

Distributed downloader workspace for WEBTOON/manhwa pages the user owns or is
authorized to archive.

Behavior:

- Uses AWS Step Functions Distributed Map over an S3 JSONL episode manifest.
- Each map item invokes `webtoon_manga_episode_worker_v2`.
- The Lambda worker tries direct source fetches first when `PROXY_MODE=auto`.
- If direct fetches fail and Decodo proxy environment variables are configured,
  the worker falls back to Decodo proxies for WEBTOON/image HTTP requests.
- S3 uploads are direct to AWS, not proxied.

Main commands:

```bash
python3 -m py_compile artifacts/webtoon_manga/*.py
python3 artifacts/webtoon_manga/generate_manifest.py
python3 artifacts/webtoon_manga/deploy_and_start.py --start --max-concurrency 1000 --proxy-mode auto
python3 artifacts/webtoon_manga/monitor.py '<execution-arn>'
```

Do not commit credentials. Pass Decodo credentials through the process
environment only if direct Lambda smoke tests fail.

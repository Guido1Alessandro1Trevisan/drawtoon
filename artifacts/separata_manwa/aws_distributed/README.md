# separata_manwa AWS Distributed Downloader

This workstation runs the authorized WEBTOON/manhwa page downloader through AWS Step Functions Distributed Map.

It stays inside `artifacts/separata_manwa/` and writes source images to:

```text
s3://drawtoon/datasets/pages/source/webtoon/
```

## Access Model

- `proxy_mode=auto` first tests direct AWS Lambda egress against the public WEBTOON list page.
- If direct access works, workers use direct web fetches and direct S3 uploads.
- If direct access fails and `--proxy-secret-name` is supplied, workers use Decodo proxies for WEBTOON fetches only.
- S3 uploads do not go through Decodo. Routing S3 writes through Decodo caused local timeout failures and is not needed.

Do not use this for locked, app-only, paywalled, DRM-protected, CAPTCHA-protected, or private API content.

## Optional Proxy Secret

If proxy fallback is needed, create an AWS Secrets Manager secret outside the repo. The secret value should be JSON:

```json
{
  "host": "dc.decodo.com",
  "ports": "10301,10302",
  "user": "example",
  "password": "example"
}
```

Do not commit or log the real secret values. The credentials previously pasted in chat should be rotated.

## Deploy And Start

From the repo root:

```bash
python3 artifacts/separata_manwa/aws_distributed/scripts/deploy_start.py \
  --region us-east-1 \
  --bucket drawtoon \
  --proxy-mode auto \
  --max-concurrency 300 \
  --worker-image-concurrency 8 \
  --tolerated-failure-count 250
```

Use `--proxy-mode always --proxy-secret-name <secret-name>` only if direct mode fails.

## Monitor

The deploy command prints an `execution_arn`. Check it with:

```bash
python3 artifacts/separata_manwa/aws_distributed/scripts/status.py '<execution_arn>'
```

For raw AWS CLI:

```bash
aws stepfunctions describe-execution --execution-arn '<execution_arn>' --region us-east-1
aws stepfunctions list-map-runs --execution-arn '<execution_arn>' --region us-east-1
```

## Tuning

Start at `--max-concurrency 300`. If failures remain low and Lambda/S3/web source are healthy, increase cautiously.

Do not start at 1000 without checking:

```bash
aws lambda get-account-settings --region us-east-1
```

If failures spike, reduce to:

```text
--max-concurrency 100 --worker-image-concurrency 4
```


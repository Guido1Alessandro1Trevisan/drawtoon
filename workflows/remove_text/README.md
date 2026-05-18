# Remove Text Workflow

Distributed manga page text removal for already-filtered Drawtoon pages.

This workflow is intentionally based on the `lineart2/text-remove-lora` base-model evaluation path:

- endpoint: `fal-ai/qwen-image-edit-2511`
- prompt: `prompts/master_prompt.md`
- steps: `40`
- guidance: `4.5`
- output: `png`
- LoRA: disabled; the workflow rejects `/lora` endpoints

Input defaults to:

```text
s3://drawtoon/datasets/pages/filtered/
```

Output defaults to:

```text
s3://drawtoon/datasets/pages/text_removed/qwen2511_master_prompt_v1/
```

Each source page gets:

- a text-removed PNG at the same relative path under the run prefix
- a status JSON under `_status/`
- Step Functions audit JSONL under `_audit/`
- job manifests/config under `_jobs/`

## Secret

Do not hardcode the fal key. Deploy with a Secrets Manager secret named by the template parameter `FalSecretName`:

```text
drawtoon-fal-key
```

The secret can be either the raw fal key or JSON containing `FAL_KEY`.

## Deploy

```bash
cd workflows/remove_text
sam build
sam deploy --guided
```

## Dry Run

```bash
python3 workflows/remove_text/start.py \
  --stack-name drawtoon-remove-text \
  --dry-run \
  remove-pages \
  --max-pages 10
```

## Launch

```bash
python3 workflows/remove_text/start.py \
  --stack-name drawtoon-remove-text \
  --job-name qwen2511-master-prompt-v1 \
  --max-concurrency 32 \
  --tolerated-failure-count 1000 \
  remove-pages
```

Use `--overwrite` only when intentionally regenerating outputs. Without it, the prepare step skips existing PNG outputs under the run prefix.

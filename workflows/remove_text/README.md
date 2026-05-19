# Drawtoon Remove Text — Modal

Local FLUX.2 Klein 9B manga text removal on Modal.

The only production model in this workflow is:

```text
black-forest-labs/FLUX.2-klein-9B
num_inference_steps = 4
variant = klein_local_9b_4step
```

No fal endpoint is used. No Qwen-Image-Edit-2511 path remains in this workflow.

The model is gated on Hugging Face. The Modal worker uses the existing
`lineart2-hf-token` secret by default, which must contain
`HF_TOKEN`. If `HF_TOKEN` is exported locally at launch time, the worker forwards
that local value as an ephemeral Modal secret instead.

## One-Time Setup

```bash
cd workflows/remove_text
modal deploy modal_klein.py
```

Weights are cached on Modal volume `flux2-klein-9b-hf-cache`.

## Bulk Annotate

Writes one PNG per source page to `s3://drawtoon/datasets/pages/text_removed/<chapter>/`,
skipping any page that already has an output there.

```bash
KLEIN_GPU=H200 KLEIN_MAX_CONTAINERS=40 python3 start.py \
  --chapters vinland-saga \
  --output-prefix datasets/pages/text_removed \
  --pages-per-shard 8 \
  --detach
```

`start.py` always checks `--output-prefix` for existing PNGs. Pass extra
`--skip-existing-prefix <path>` flags to union in additional prefixes (e.g. an
older one-off run you don't want to redo).

## Files

```text
workflows/remove_text/
├── README.md
├── start.py
├── modal_klein.py
└── prompts/master_prompt.md
```

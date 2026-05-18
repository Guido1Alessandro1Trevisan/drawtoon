# Current Modal Runs 2

Snapshot: 2026-05-15 22:50 CEST (+0200)

Source: `modal app list` from the `reinforcenow/main` workspace.

## Active Training Apps

| App ID | State | Tasks | Job | Notes |
|---|---|---:|---|---|
| `ap-YFKnpoUX5rXj0NxH3KysHe` | detached | 2 | `mangazero_flux2_klein9b_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga8` | General MangaZero panel prediction, not a 1000/60-panel slice |
| `ap-U9utV8U2LD4m2LccirO3XC` | detached | 2 | `mangazero_flux2_klein9b_attack_on_titan_1000panels_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga1_h200` | Attack-on-Titan 1000-panel slice, rank 32 |
| `ap-kfAyKsh3iUv7BTUgv1RmSO` | detached | 2 | `mangazero_flux2_klein9b_attack_on_titan_prompt1_flat_caption_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga1_h200` | Attack-on-Titan prompt1 ablation, flat caption text |
| `ap-G3zsDdBz1eQjuHrXQMOuTG` | detached | 2 | `mangazero_flux2_klein9b_attack_on_titan_prompt2_json_structured_caption_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga1_h200` | Attack-on-Titan prompt2 ablation, structured JSON caption |
| `ap-R8Jk3fa2RKLTxcut03zsvq` | detached | 110 | `mangazero_flux2_klein9b_mangazero_full_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga1_h200_1epoch` | Full MangaZero one-epoch run on H200:8 |
| `ap-EVmRH4js3H8fdI7uocOTlv` | detached | 2 | `mangazero_flux2_klein9b_mangazero_full_same_page_refs_native_pad16_haiku45_lora_r128_lr5e5_ga1_h200_1epoch` | Full MangaZero one-epoch run on H200:8, rank 128 LoRA |

## Stopped Runs

| App ID | Job |
|---|---|
| `ap-dS7QkZwaU95O2pLD9XJZMT` | `mangazero_flux2_klein9b_mangazero_full_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga1_h200_1epoch` |
| `ap-7mKPO5yvyP9GQYJ9Tl1YBl` | previous full-dataset launcher attempt |
| `ap-uMjtpw58uqayu9TezY0iZY` | previous full-dataset launcher attempt |
| `ap-2HHiLWGDLHzK2kuxFv0MmK` | Attack-on-Titan 60-panel slice, rank 32 |
| `ap-dfVScBxA7A3Sv8Bt2JKTxD` | prompt ablation launcher attempt |
| `ap-3ZGE6vTfFoyqXWednwEPso` | prompt ablation launcher attempt |

## Notes

- `ap-R8Jk3fa2RKLTxcut03zsvq` is the live full-dataset H200:8 run.
- The prompt ablation apps are the current prompt1 / prompt2 experiments.
- `ap-U9utV8U2LD4m2LccirO3XC` is the 1000-panel Attack-on-Titan slice.
- `ap-2HHiLWGDLHzK2kuxFv0MmK` was the 60-panel Attack-on-Titan slice and is stopped.
- The older Attack on Titan slice runs are stopped.

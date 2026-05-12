import re
from dataclasses import dataclass
from typing import Any, Callable

import torch
from PIL import Image
from torch import Tensor

MAX_GIA_CHARACTER_REFS = 6
MAX_GIA_DIALOGUE_REGIONS = 10


@dataclass
class GIAPromptSpans:
    prompt: str
    sad_token_indices: list[Tensor]
    cei_token_indices: Tensor
    empty_prompt_token_indices: Tensor


def infer_gia_ref_indices(inputs: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for key in inputs:
        match = re.fullmatch(r"ref_img_(\d+)", str(key))
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def normalize_gia_sad(entity: dict[str, Any]) -> str:
    sad = entity.get("SAD", "")
    if isinstance(sad, dict) and "id" in sad and "desc" in sad:
        sad = f"{sad['id']},{sad['desc']}."
    return str(sad)


def build_gia_prompt(inputs: dict[str, Any]) -> tuple[str, list[str], str]:
    ref_indices = infer_gia_ref_indices(inputs)
    if not ref_indices:
        raise ValueError("gia_inputs requires at least one ref_img_N entry")
    for index in ref_indices:
        if not isinstance(inputs.get(f"ref_img_{index}"), dict):
            raise ValueError(f"gia_inputs ref_img_{index} must be an object")
    sad_segments = [
        normalize_gia_sad(inputs[f"ref_img_{index}"])
        for index in ref_indices
    ]
    for index, sad in zip(ref_indices, sad_segments):
        if not sad.strip():
            raise ValueError(f"gia_inputs ref_img_{index} requires a non-empty SAD")
    cei = str(inputs.get("CEI", ""))
    if not cei.strip():
        raise ValueError("gia_inputs requires a non-empty CEI")
    return "".join(sad_segments) + cei, sad_segments, cei


def build_gia_inputs_from_lamic_sample(
    sample: dict[str, Any],
    *,
    resolve_image_path: Callable[[str], Any] | None = None,
) -> dict[str, Any] | None:
    explicit_inputs = sample.get("gia_inputs")
    if isinstance(explicit_inputs, dict):
        return cap_gia_inputs(explicit_inputs)

    lamic = sample.get("lamic")
    if not isinstance(lamic, dict):
        return None
    references = lamic.get("references")
    if not isinstance(references, list) or not references:
        return None

    cei = str(lamic.get("CEI") or sample.get("caption") or "").strip()
    if not cei:
        return None

    gia_inputs: dict[str, Any] = {"CEI": cei}
    next_index = 1
    for reference in references:
        if not isinstance(reference, dict):
            continue
        sad = reference.get("SAD")
        bbox = reference.get("target_box_norm") or reference.get("bbox")
        if sad is None or bbox is None:
            continue

        image_path = reference.get("image_path")
        if image_path and resolve_image_path is not None:
            image_path = resolve_image_path(str(image_path))

        gia_inputs[f"ref_img_{next_index}"] = {
            "image_path": image_path,
            "SAD": sad,
            "bbox": bbox,
        }
        next_index += 1

    if next_index == 1:
        return None
    return cap_gia_inputs(gia_inputs)


def has_gia_reference_image(entity: dict[str, Any]) -> bool:
    image_path = entity.get("image_path")
    if image_path is None:
        return False
    image_path = str(image_path).strip()
    return bool(image_path) and image_path.lower() not in {"none", "null"}


def cap_gia_inputs(
    inputs: dict[str, Any],
    *,
    max_character_refs: int = MAX_GIA_CHARACTER_REFS,
    max_dialogue_regions: int = MAX_GIA_DIALOGUE_REGIONS,
) -> dict[str, Any]:
    max_character_refs = max(0, int(max_character_refs))
    max_dialogue_regions = max(0, int(max_dialogue_regions))
    kept: list[dict[str, Any]] = []
    character_count = 0
    dialogue_count = 0
    for index in infer_gia_ref_indices(inputs):
        entity = inputs.get(f"ref_img_{index}")
        if not isinstance(entity, dict):
            continue
        if has_gia_reference_image(entity):
            if character_count >= max_character_refs:
                continue
            character_count += 1
        else:
            if dialogue_count >= max_dialogue_regions:
                continue
            dialogue_count += 1
        kept.append(dict(entity))

    capped = {"CEI": inputs.get("CEI", "")}
    for new_index, entity in enumerate(kept, start=1):
        capped[f"ref_img_{new_index}"] = entity
    return capped


def load_gia_reference_images(inputs: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for index in infer_gia_ref_indices(inputs):
        entity = inputs[f"ref_img_{index}"]
        if not has_gia_reference_image(entity):
            continue
        images.append(Image.open(str(entity["image_path"])).convert("RGB"))
    return images


def _box_from_gia_entity(entity: dict[str, Any], *, ref_index: int) -> tuple[float, float, float, float] | None:
    box = entity.get("bbox")
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"gia_inputs ref_img_{ref_index} bbox must be [x0, y0, x1, y1]")
    try:
        x0, y0, x1, y1 = [float(value) for value in box]
    except (TypeError, ValueError):
        raise ValueError(f"gia_inputs ref_img_{ref_index} bbox must contain numeric corner coordinates")
    if not all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)):
        raise ValueError(f"gia_inputs ref_img_{ref_index} bbox must be normalized [x0, y0, x1, y1]")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"gia_inputs ref_img_{ref_index} bbox must have positive area")
    return x0, y0, x1, y1


def build_gia_prompt_spans_from_offsets(
    inputs: dict[str, Any],
    *,
    offsets: Tensor,
    attention_mask: Tensor,
    prompt_start: int,
    max_sequence_length: int,
) -> GIAPromptSpans:
    prompt, sad_segments, cei = build_gia_prompt(inputs)
    assigned = torch.zeros(max_sequence_length, dtype=torch.bool)
    sad_token_indices: list[Tensor] = []
    cursor = int(prompt_start)
    attention_mask = attention_mask.to(torch.bool).cpu()
    offsets = offsets.cpu()

    for segment in sad_segments:
        start = cursor
        end = cursor + len(segment)
        token_indices = torch.where(
            attention_mask & (offsets[:, 1] > start) & (offsets[:, 0] < end)
        )[0]
        sad_token_indices.append(token_indices)
        if token_indices.numel() > 0:
            assigned[token_indices] = True
        cursor = end

    cei_start = cursor
    cei_end = cursor + len(cei)
    cei_token_indices = torch.where(
        attention_mask & (offsets[:, 1] > cei_start) & (offsets[:, 0] < cei_end)
    )[0]
    if cei_token_indices.numel() > 0:
        assigned[cei_token_indices] = True

    empty_prompt_token_indices = torch.where(~assigned)[0]
    return GIAPromptSpans(
        prompt=prompt,
        sad_token_indices=sad_token_indices,
        cei_token_indices=cei_token_indices,
        empty_prompt_token_indices=empty_prompt_token_indices,
    )


def target_region_token_indices(
    img_ids: Tensor,
    inputs: dict[str, Any],
    *,
    num_txt_tokens: int,
    target_token_count: int,
) -> tuple[list[Tensor], Tensor]:
    target_ids = img_ids[:target_token_count].to(torch.long).cpu()
    if target_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return [empty for _ in infer_gia_ref_indices(inputs)], empty

    h_ids = target_ids[:, 1]
    w_ids = target_ids[:, 2]
    grid_h = int(h_ids.max().item()) + 1
    grid_w = int(w_ids.max().item()) + 1
    all_target = torch.arange(target_token_count, dtype=torch.long)
    covered = torch.zeros(target_token_count, dtype=torch.bool)
    groups: list[Tensor] = []

    for index in infer_gia_ref_indices(inputs):
        box = _box_from_gia_entity(inputs[f"ref_img_{index}"], ref_index=index)
        if box is None:
            local_indices = all_target
        else:
            x0, y0, x1, y1 = box
            token_x0 = w_ids.to(torch.float32) / float(grid_w)
            token_x1 = (w_ids.to(torch.float32) + 1.0) / float(grid_w)
            token_y0 = h_ids.to(torch.float32) / float(grid_h)
            token_y1 = (h_ids.to(torch.float32) + 1.0) / float(grid_h)
            overlap_w = torch.clamp(
                torch.minimum(token_x1, torch.tensor(x1)) - torch.maximum(token_x0, torch.tensor(x0)),
                min=0.0,
            )
            overlap_h = torch.clamp(
                torch.minimum(token_y1, torch.tensor(y1)) - torch.maximum(token_y0, torch.tensor(y0)),
                min=0.0,
            )
            local_indices = torch.where((overlap_w * overlap_h) > 0.0)[0]
            if local_indices.numel() == 0:
                raise ValueError(f"gia_inputs ref_img_{index} bbox does not cover any target image tokens")
        covered[local_indices] = True
        groups.append(local_indices + num_txt_tokens)

    uncontrolled = torch.where(~covered)[0] + num_txt_tokens
    return groups, uncontrolled


def reference_token_indices(
    img_ids: Tensor,
    inputs: dict[str, Any],
    *,
    num_txt_tokens: int,
    target_token_count: int,
) -> list[Tensor]:
    ref_indices = infer_gia_ref_indices(inputs)
    empty = torch.empty(0, dtype=torch.long)
    if img_ids.shape[0] <= target_token_count:
        return [empty for _ in ref_indices]

    ref_ids = img_ids[target_token_count:].to(torch.long).cpu()
    ref_t_values = torch.unique(ref_ids[:, 0], sorted=True)
    ref_groups_by_t: list[Tensor] = []
    ref_base = num_txt_tokens + target_token_count
    for t_value in ref_t_values:
        local = torch.where(ref_ids[:, 0] == t_value)[0] + ref_base
        ref_groups_by_t.append(local)

    output: list[Tensor] = []
    next_ref_group = 0
    for index in ref_indices:
        entity = inputs[f"ref_img_{index}"]
        if has_gia_reference_image(entity):
            output.append(
                ref_groups_by_t[next_ref_group]
                if next_ref_group < len(ref_groups_by_t)
                else empty
            )
            next_ref_group += 1
        else:
            output.append(empty)
    return output


def build_gia_attention_mask(
    inputs: dict[str, Any],
    prompt_spans: GIAPromptSpans,
    *,
    img_ids: Tensor,
    num_txt_tokens: int,
    target_token_count: int,
) -> Tensor:
    ref_indices = infer_gia_ref_indices(inputs)
    region_groups, uncontrolled_region = target_region_token_indices(
        img_ids,
        inputs,
        num_txt_tokens=num_txt_tokens,
        target_token_count=target_token_count,
    )
    ref_groups = reference_token_indices(
        img_ids,
        inputs,
        num_txt_tokens=num_txt_tokens,
        target_token_count=target_token_count,
    )
    total_tokens = int(num_txt_tokens + img_ids.shape[0])
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)

    def cat_nonempty(groups: list[Tensor]) -> Tensor:
        nonempty = [group for group in groups if group.numel() > 0]
        if not nonempty:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(nonempty, dim=0)

    sad_groups = prompt_spans.sad_token_indices
    cei = prompt_spans.cei_token_indices
    empty_prompt = prompt_spans.empty_prompt_token_indices
    all_regions = cat_nonempty(region_groups)
    all_refs = cat_nonempty(ref_groups)

    def allow(query: Tensor, key: Tensor):
        if query.numel() == 0 or key.numel() == 0:
            return
        rows, cols = torch.meshgrid(query.to(torch.long), key.to(torch.long), indexing="ij")
        mask[rows, cols] = True

    all_tokens = torch.arange(total_tokens, dtype=torch.long)
    allow(empty_prompt, all_tokens)
    allow(all_tokens, empty_prompt)

    allow(cei, torch.cat([*sad_groups, cei, all_regions, uncontrolled_region, all_refs]))
    allow(torch.cat([*sad_groups, cei, all_regions, uncontrolled_region, all_refs]), cei)

    for group_idx, _ in enumerate(ref_indices):
        sad = sad_groups[group_idx] if group_idx < len(sad_groups) else torch.empty(0, dtype=torch.long)
        region = region_groups[group_idx]
        ref = ref_groups[group_idx]
        allow(sad, sad)
        allow(region, region)
        allow(ref, ref)
        allow(sad, region)
        allow(region, sad)
        allow(sad, ref)
        allow(ref, sad)
        allow(region, ref)
        allow(ref, region)

    for region in region_groups:
        allow(region, all_regions)
        allow(region, uncontrolled_region)
    allow(uncontrolled_region, uncontrolled_region)
    allow(uncontrolled_region, all_regions)

    mask.fill_diagonal_(True)
    return mask


def bool_mask_to_additive_attention_mask(mask: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    mask = mask.to(device=device)
    if mask.dtype == torch.bool:
        mask = torch.where(
            mask,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), -torch.finfo(dtype).max, device=device, dtype=dtype),
        )
    else:
        mask = mask.to(dtype=dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask

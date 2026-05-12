import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FLUX2_SRC = ROOT / "extensions_built_in" / "diffusion_models" / "flux2" / "src"

package = types.ModuleType("flux2_test")
package.__path__ = [str(FLUX2_SRC)]
sys.modules.setdefault("flux2_test", package)

model_spec = importlib.util.spec_from_file_location(
    "flux2_test.model",
    FLUX2_SRC / "model.py",
)
model_module = importlib.util.module_from_spec(model_spec)
sys.modules["flux2_test.model"] = model_module
model_spec.loader.exec_module(model_module)

gia_spec = importlib.util.spec_from_file_location(
    "flux2_test.gia",
    FLUX2_SRC / "gia.py",
)
gia_module = importlib.util.module_from_spec(gia_spec)
sys.modules["flux2_test.gia"] = gia_module
gia_spec.loader.exec_module(gia_module)

Flux2 = model_module.Flux2
Flux2Params = model_module.Flux2Params


def _gia_inputs(image_path: str | None = "ref.png") -> dict:
    return {
        "ref_img_1": {
            "image_path": image_path,
            "SAD": {"id": "char_1", "desc": "keep the same face and hair"},
            "bbox": [0.0, 0.0, 0.5, 0.5],
        },
        "ref_img_2": {
            "image_path": None,
            "SAD": "white manga speech bubble for char_1.",
            "bbox": [0.5, 0.0, 1.0, 0.5],
        },
        "CEI": "black and white manga panel.",
    }


def _char_offsets(text: str, max_sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.zeros((max_sequence_length, 2), dtype=torch.long)
    attention_mask = torch.zeros((max_sequence_length,), dtype=torch.long)
    for idx in range(min(len(text), max_sequence_length)):
        offsets[idx] = torch.tensor([idx, idx + 1])
        attention_mask[idx] = 1
    return offsets, attention_mask


def test_gia_prompt_builds_sads_then_cei_and_token_spans():
    inputs = _gia_inputs()
    prompt, sad_segments, cei = gia_module.build_gia_prompt(inputs)

    assert sad_segments == [
        "char_1,keep the same face and hair.",
        "white manga speech bubble for char_1.",
    ]
    assert cei == "black and white manga panel."
    assert prompt == "".join(sad_segments) + cei

    offsets, attention_mask = _char_offsets(prompt, len(prompt) + 4)
    spans = gia_module.build_gia_prompt_spans_from_offsets(
        inputs,
        offsets=offsets,
        attention_mask=attention_mask,
        prompt_start=0,
        max_sequence_length=len(prompt) + 4,
    )

    assert torch.equal(spans.sad_token_indices[0], torch.arange(0, len(sad_segments[0])))
    assert torch.equal(
        spans.sad_token_indices[1],
        torch.arange(len(sad_segments[0]), len(sad_segments[0]) + len(sad_segments[1])),
    )
    assert torch.equal(
        spans.cei_token_indices,
        torch.arange(len(sad_segments[0]) + len(sad_segments[1]), len(prompt)),
    )


def test_gia_requires_sad_and_cei():
    missing_cei = _gia_inputs()
    missing_cei["CEI"] = ""
    with pytest.raises(ValueError, match="CEI"):
        gia_module.build_gia_prompt(missing_cei)

    missing_sad = _gia_inputs()
    missing_sad["ref_img_1"]["SAD"] = ""
    with pytest.raises(ValueError, match="SAD"):
        gia_module.build_gia_prompt(missing_sad)


def test_gia_inputs_can_be_built_from_lamic_manifest_sample():
    sample = {
        "caption": "fallback panel caption",
        "lamic": {
            "CEI": "global instruction",
            "references": [
                {
                    "image_path": "s3://bucket/ref.png",
                    "SAD": {"id": "char_1", "desc": "same character"},
                    "target_box_norm": [0.1, 0.2, 0.3, 0.4],
                },
                {
                    "SAD": "white speech bubble.",
                    "bbox": [0.5, 0.1, 0.8, 0.3],
                },
            ],
        },
    }

    inputs = gia_module.build_gia_inputs_from_lamic_sample(
        sample,
        resolve_image_path=lambda path: f"/mnt/{path.removeprefix('s3://')}",
    )
    prompt, sad_segments, cei = gia_module.build_gia_prompt(inputs)

    assert inputs["CEI"] == "global instruction"
    assert inputs["ref_img_1"]["image_path"] == "/mnt/bucket/ref.png"
    assert inputs["ref_img_1"]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert inputs["ref_img_2"]["image_path"] is None
    assert sad_segments == ["char_1,same character.", "white speech bubble."]
    assert cei == "global instruction"
    assert prompt == "char_1,same character.white speech bubble.global instruction"


def test_gia_bbox_regions_refs_and_cei_permissions_are_wired():
    inputs = _gia_inputs()
    prompt, _, _ = gia_module.build_gia_prompt(inputs)
    offsets, attention_mask = _char_offsets(prompt, len(prompt))
    spans = gia_module.build_gia_prompt_spans_from_offsets(
        inputs,
        offsets=offsets,
        attention_mask=attention_mask,
        prompt_start=0,
        max_sequence_length=len(prompt),
    )
    num_txt_tokens = len(prompt)
    target_token_count = 4
    img_ids = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [10, 0, 0, 0],
            [10, 0, 1, 0],
        ],
        dtype=torch.long,
    )

    region_groups, uncontrolled = gia_module.target_region_token_indices(
        img_ids,
        inputs,
        num_txt_tokens=num_txt_tokens,
        target_token_count=target_token_count,
    )
    ref_groups = gia_module.reference_token_indices(
        img_ids,
        inputs,
        num_txt_tokens=num_txt_tokens,
        target_token_count=target_token_count,
    )
    mask = gia_module.build_gia_attention_mask(
        inputs,
        spans,
        img_ids=img_ids,
        num_txt_tokens=num_txt_tokens,
        target_token_count=target_token_count,
    )

    sad1 = spans.sad_token_indices[0][0]
    sad2 = spans.sad_token_indices[1][0]
    cei = spans.cei_token_indices[0]
    reg1 = region_groups[0][0]
    reg2 = region_groups[1][0]
    ureg = uncontrolled[0]
    ref1 = ref_groups[0][0]

    assert torch.equal(region_groups[0], torch.tensor([num_txt_tokens]))
    assert torch.equal(region_groups[1], torch.tensor([num_txt_tokens + 1]))
    assert torch.equal(uncontrolled, torch.tensor([num_txt_tokens + 2, num_txt_tokens + 3]))
    assert torch.equal(ref_groups[0], torch.tensor([num_txt_tokens + 4, num_txt_tokens + 5]))
    assert ref_groups[1].numel() == 0

    assert mask[sad1, reg1]
    assert not mask[sad1, reg2]
    assert mask[sad2, reg2]
    assert not mask[sad2, reg1]
    assert mask[reg1, ref1]
    assert not mask[reg2, ref1]
    assert mask[cei, reg1] and mask[cei, reg2] and mask[cei, ref1] and mask[cei, ureg]
    assert not mask[ureg, sad1]
    assert mask[ureg, reg1] and mask[reg1, ureg]


def test_gia_attention_mask_allows_text_only_groups_without_ref_tokens():
    inputs = _gia_inputs(image_path=None)
    prompt, _, _ = gia_module.build_gia_prompt(inputs)
    offsets, attention_mask = _char_offsets(prompt, len(prompt))
    spans = gia_module.build_gia_prompt_spans_from_offsets(
        inputs,
        offsets=offsets,
        attention_mask=attention_mask,
        prompt_start=0,
        max_sequence_length=len(prompt),
    )
    num_txt_tokens = len(prompt)
    target_token_count = 4
    img_ids = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    ref_groups = gia_module.reference_token_indices(
        img_ids,
        inputs,
        num_txt_tokens=num_txt_tokens,
        target_token_count=target_token_count,
    )
    mask = gia_module.build_gia_attention_mask(
        inputs,
        spans,
        img_ids=img_ids,
        num_txt_tokens=num_txt_tokens,
        target_token_count=target_token_count,
    )

    assert all(group.numel() == 0 for group in ref_groups)
    assert mask.shape == (num_txt_tokens + target_token_count, num_txt_tokens + target_token_count)
    assert mask[spans.cei_token_indices[0], num_txt_tokens]


def test_gia_bboxes_are_normalized_xyxy_corners_and_use_token_overlap():
    inputs = _gia_inputs()
    inputs["ref_img_1"]["bbox"] = [0.49, 0.0, 0.51, 0.5]
    inputs["ref_img_2"]["bbox"] = None
    num_txt_tokens = 8
    img_ids = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    region_groups, uncontrolled = gia_module.target_region_token_indices(
        img_ids,
        inputs,
        num_txt_tokens=num_txt_tokens,
        target_token_count=4,
    )

    assert torch.equal(region_groups[0], torch.tensor([num_txt_tokens, num_txt_tokens + 1]))
    assert torch.equal(
        region_groups[1],
        torch.tensor([num_txt_tokens, num_txt_tokens + 1, num_txt_tokens + 2, num_txt_tokens + 3]),
    )
    assert uncontrolled.numel() == 0


def test_gia_rejects_invalid_or_pixel_bboxes_instead_of_clamping():
    inputs = _gia_inputs()
    img_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)

    inputs["ref_img_1"]["bbox"] = [0, 0, 2, 1]
    with pytest.raises(ValueError, match="normalized"):
        gia_module.target_region_token_indices(
            img_ids,
            inputs,
            num_txt_tokens=4,
            target_token_count=1,
        )

    inputs["ref_img_1"]["bbox"] = [0.5, 0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="positive area"):
        gia_module.target_region_token_indices(
            img_ids,
            inputs,
            num_txt_tokens=4,
            target_token_count=1,
        )


def test_gia_additive_mask_shape_and_ref_image_loading(tmp_path):
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (2, 2), "white").save(ref_path)
    inputs = _gia_inputs(str(ref_path))

    refs = gia_module.load_gia_reference_images(inputs)
    assert len(refs) == 1
    assert refs[0].size == (2, 2)

    additive = gia_module.bool_mask_to_additive_attention_mask(
        torch.tensor([[True, False], [False, True]]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert additive.shape == (1, 1, 2, 2)
    assert additive[0, 0, 0, 0].item() == 0.0
    assert additive[0, 0, 0, 1].item() < -1e20


def test_flux2_accepts_gia_attention_mask_and_has_no_old_layout_branch():
    torch.manual_seed(0)
    model = Flux2(
        Flux2Params(
            in_channels=4,
            context_in_dim=12,
            hidden_size=16,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=[2, 2, 2, 2],
            mlp_ratio=1.0,
            use_guidance_embed=False,
        )
    )
    model.eval()
    assert not hasattr(model, "text_" + "layout_type_embeddings")

    x = torch.randn(1, 2, 4)
    ctx = torch.randn(1, 3, 12)
    x_ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 0]]])
    ctx_ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]])
    timestep = torch.tensor([0.5])
    all_allowed = gia_module.bool_mask_to_additive_attention_mask(
        torch.ones((5, 5), dtype=torch.bool),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.no_grad():
        unmasked = model(
            x=x,
            x_ids=x_ids,
            timesteps=timestep,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
        )
        masked = model(
            x=x,
            x_ids=x_ids,
            timesteps=timestep,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
            attention_mask=all_allowed,
        )

    assert unmasked.shape == (1, 2, 4)
    assert torch.allclose(unmasked, masked, atol=1e-5, rtol=1e-5)

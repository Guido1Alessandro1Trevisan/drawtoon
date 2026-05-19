from logging import INFO, Logger
from typing import Any, List, Union

import torch
import torch.nn as nn
import json
from layousyn.config import Config
from layousyn.diffusion.gaussian_diffusion import GaussianDiffusion
from layousyn.utils import find_model
from layousyn.model.dit import LDiT_models
from layousyn.prompt_handler.gpt_object_count_handler import GPTObjectCountPromptHandler
from layousyn.prompt_handler.llama_object_count_handler import LlamaObjectCountPromptHandler
from layousyn.prompt_handler.qwen_object_count_handler import QwenObjectCountPromptHandler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_json(fname):
    with open(fname, "r") as file:
        data = json.load(file)
    return data

@torch.no_grad()
def encode_labels(
    labels_set: List[List[str]],
    encoder: Any,
    config: Config,
    device: Union[str, int] = "cuda",
):
    # get unique labels and their encodings
    unique_labels = [""]
    for labels in labels_set:
        unique_labels += labels
    unique_labels = list(set(unique_labels))
    unique_idx_label_map = {obj: i for i, obj in enumerate(unique_labels)}
    unique_labels_encoded = torch.tensor(encoder.encode(unique_labels), device=device)

    # pad labels
    labels_set_padded = []
    for labels in labels_set:
        labels_set_padded.append(labels + [""] * (config.max_in_len - len(labels)))

    # get labels encodings
    x_enc = torch.zeros(
        len(labels_set_padded),
        config.max_in_len,
        config.concept_in_channel,
        device=device,
    )
    for i, labels in enumerate(labels_set_padded):
        labels = labels[: config.max_in_len]
        x_enc[i, :, :] = unique_labels_encoded[
            [unique_idx_label_map[label] for label in labels]
        ]
    x_enc = x_enc.detach().float()

    # get padding mask
    x_padding_mask = torch.tensor(
        [
            [False if label != "" else True for label in labels[: config.max_in_len]]
            for labels in labels_set_padded
        ],
        device=device,
    )

    return x_enc, x_padding_mask


@torch.no_grad()
def encode_captions(captions: List[str], encoder: Any, batch_size: int = 512):
    y = []
    y_mask = []

    # encode captions in batches
    for i in range(0, len(captions), batch_size):
        _, y_batch, y_padding_mask = encoder.get_text_embeddings(
            captions[i : i + batch_size]
        )
        y.append(y_batch)
        y_mask.append(y_padding_mask)

    # concatenate batches
    y = torch.cat(y, 0).detach().float()
    y_mask = torch.cat(y_mask, 0).detach()

    return y, y_mask


def get_sampling_type(type: str, diffusion: GaussianDiffusion):
    if type == "ddpm":
        print(f"Sampling with type: p_sample_loop")
        return diffusion.p_sample_loop
    elif type == "ddim":
        print(f"Sampling with type: ddim_sample_loop")
        return diffusion.ddim_sample_loop
    else:
        raise ValueError(f"Invalid type: {type}")


@torch.no_grad()
def sample(
    x,
    x_padding_mask,
    model_kwargs: dict,
    sample_fn: Any,
    diffusion: GaussianDiffusion,
    device: Union[str, int] = "cuda",
    type: str = "ddim",
    return_samples: bool = False,
):
    type_fn = get_sampling_type(type, diffusion)
    output_bboxs = type_fn(
        sample_fn,
        x.shape,
        x,
        x_padding_mask=x_padding_mask,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        return_samples=return_samples,
        device=device,
    )
    output_bboxs = torch.clamp(output_bboxs, -1.0, 1.0)
    return output_bboxs.detach().cpu()


@torch.no_grad()
def sample_with_cfg(
    captions: List[str],
    labels_set: List[List[str]],
    config: Config,
    diffusion: GaussianDiffusion,
    model: nn.Module,
    label_encoder: Any,
    caption_encoder: Any,
    cfg_scale: float = 8.0,
    ar_int=1.0,
    device: Union[str, int] = "cuda",
    type: str = "ddim",
    return_samples: bool = False,
) -> torch.Tensor:
    
    # batch size
    batch_size = len(captions)

    # setup labels encoding
    x = torch.randn(batch_size, config.max_in_len, config.in_channel, device=device)
    x_enc, x_padding_mask = encode_labels(labels_set, label_encoder, config, device)
    y, y_padding_mask = encode_captions(captions, caption_encoder)
    ar = torch.tensor([ar_int] * batch_size, device=device).float()

    # Setup classifier-free guidance:
    x = torch.cat([x, x], 0)
    x_enc, x_padding_mask = torch.cat([x_enc, x_enc], 0), torch.cat(
        [x_padding_mask, x_padding_mask], 0
    )
    y_null = model.y_embedder.y_embedding.to(device).repeat(batch_size, 1, 1)
    y_mask_null = model.y_embedder.y_padding_mask.to(device).repeat(batch_size, 1)
    y, y_padding_mask = torch.cat([y, y_null], 0), torch.cat([y_padding_mask, y_mask_null], 0)
    ar = torch.cat([ar, ar], 0)

    # run denoiser
    model_kwargs = dict(
        ar=ar, x_enc=x_enc, y=y, y_padding_mask=y_padding_mask, cfg_scale=cfg_scale
    )
    output_bboxs = sample(
        x=x,
        x_padding_mask=x_padding_mask,
        model_kwargs=model_kwargs,
        sample_fn=model.forward_with_cfg,
        diffusion=diffusion,
        device=device,
        type=type,
        return_samples=return_samples,
    )
    output_bboxs = output_bboxs[:batch_size]

    return output_bboxs


@torch.no_grad()
def invert(
    x,
    x_padding_mask,
    num_steps,
    model_kwargs: dict,
    sample_fn: Any,
    diffusion: GaussianDiffusion,
):
    timesteps = list(range(diffusion.num_timesteps))
    assert num_steps <= len(timesteps)
    print(f"Inverting for {num_steps} steps")
    for i in tqdm(range(1, num_steps), total=num_steps - 1):
        if i >= num_steps - 1:
            continue
        t = torch.tensor([timesteps[i]] * x.shape[0], device=x.device)
        x_prev = diffusion.ddim_reverse_sample(
            sample_fn,
            x,
            x_padding_mask=x_padding_mask,
            t=t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )["sample"]
        x = x_prev
    return x
    

@torch.no_grad()
def sample_with_init(
    captions: List[str],
    labels_set: List[List[str]],
    bboxes: List[List[List[float]]],
    config: Config,
    diffusion: GaussianDiffusion,
    model: nn.Module,
    label_encoder: Any,
    caption_encoder: Any,
    invert_steps: int,
    cfg_scale: float = 8.0,
    ar_int=1.0,
    device: Union[str, int] = "cuda",
    type: str = "ddim",
    return_samples: bool = False,
):
    batch_size = len(captions)
    x = [torch.as_tensor(bbox).reshape(-1, config.in_channel).float() for bbox in bboxes]
    x = torch.nested.as_nested_tensor(x)
    x = x.to_padded_tensor(0, (batch_size, config.max_in_len, config.in_channel)).to(device)

    x_enc, x_padding_mask = encode_labels(labels_set, label_encoder, config, device)
    y, y_mask = encode_captions(captions, caption_encoder)
    ar = torch.tensor([ar_int] * batch_size, device=device).float()

    # Setup classifier-free guidance:
    x = torch.cat([x, x], 0)
    x_enc, x_padding_mask = torch.cat([x_enc, x_enc], 0), torch.cat(
        [x_padding_mask, x_padding_mask], 0
    )
    y_null = model.y_embedder.y_embedding.to(device).repeat(batch_size, 1, 1)
    y_mask_null = model.y_embedder.y_padding_mask.to(device).repeat(batch_size, 1)
    y, y_mask = torch.cat([y, y_null], 0), torch.cat([y_mask, y_mask_null], 0)
    ar = torch.cat([ar, ar], 0)

    model_kwargs = dict(
        ar=ar, x_enc=x_enc, y=y, y_padding_mask=y_mask, cfg_scale=cfg_scale
    )

    # Do DDIM inversion
    x_inv = invert(
        x,
        x_padding_mask=x_padding_mask,
        num_steps=invert_steps,
        model_kwargs=model_kwargs,
        sample_fn=model.forward_with_cfg,
        diffusion=diffusion,
    )

    original_num_steps = diffusion.num_timesteps
    # Hack impl to reduce number of steps
    diffusion.num_timesteps = invert_steps

    # Run denoiser
    output_bboxs = sample(
        x=x_inv,
        x_padding_mask=x_padding_mask,
        model_kwargs=model_kwargs,
        sample_fn=model.forward_with_cfg,
        diffusion=diffusion,
        device=device,
        type=type,
        return_samples=return_samples,
    )
    output_bboxs = output_bboxs[:batch_size]

    diffusion.num_timesteps = original_num_steps

    return output_bboxs


@torch.no_grad()
def sample_conditional(
    captions: List[str],
    labels_set: List[List[str]],
    config: Config,
    diffusion: GaussianDiffusion,
    model: nn.Module,
    label_encoder: Any,
    caption_encoder: Any,
    ar_int=1.0,
    device: Union[str, int] = "cuda",
    type: str = "ddim",
    return_samples: bool = False,
):
    batch_size = len(captions)

    # setup labels encoding
    x = torch.randn(batch_size, config.max_in_len, config.in_channel, device=device)
    x_enc, x_padding_mask = encode_labels(labels_set, label_encoder, config, device)
    y, y_mask = encode_captions(captions, caption_encoder)
    ar = torch.tensor([ar_int] * batch_size, device=device).float()

    # run denoiser
    model_kwargs = dict(ar=ar, x_enc=x_enc, y=y, y_padding_mask=y_mask)
    output_bboxs = sample(
        x=x,
        x_padding_mask=x_padding_mask,
        model_kwargs=model_kwargs,
        sample_fn=model,
        diffusion=diffusion,
        device=device,
        type=type,
        return_samples=return_samples,
    )

    return output_bboxs


@torch.no_grad()
def sample_unconditional(
    labels_set: List[List[str]],
    config: Config,
    diffusion: GaussianDiffusion,
    model: nn.Module,
    label_encoder: Any,
    ar_int=1.0,
    type: str = "ddim",
    device: Union[str, int] = "cuda",
    return_samples: bool = False,
):
    """
    Sample from the model without any conditioning.

    Args:
        labels_set: List of labels to condition on.
        config: Config object.
        diffusion: GaussianDiffusion object.
        model: nn.Module object.
        label_encoder: SentenceTransformer object.
        ar_int: Autoregressive parameter.
        device: Device to run on.
        return_samples: Whether to return samples.

    Returns:
        output_bboxs: List of output bounding boxes.

    """
    batch_size = len(labels_set)

    # setup labels encoding
    x = torch.randn(batch_size, config.max_in_len, config.in_channel, device=device)
    x_enc, x_padding_mask = encode_labels(labels_set, label_encoder, config, device)
    y = model.y_embedder.y_embedding.to(device).repeat(batch_size, 1, 1).float()
    y_mask = model.y_embedder.y_padding_mask.to(device).repeat(batch_size, 1)
    ar = torch.tensor([ar_int] * batch_size, device=device).float()

    # run denoiser
    model_kwargs = dict(ar=ar, x_enc=x_enc, y=y, y_padding_mask=y_mask)
    output_bboxs = sample(
        x=x,
        x_padding_mask=x_padding_mask,
        model_kwargs=model_kwargs,
        sample_fn=model,
        diffusion=diffusion,
        device=device,
        type=type,
        return_samples=return_samples,
    )

    return output_bboxs


def load_model(
    ckpt: str,
    config: Config,
    device: Union[str, int] = "cuda",
    logger: Logger = Logger(name="load_model", level=0),
):
    model = LDiT_models(logger)[config.model](
        in_channels=config.in_channel,
        concept_in_channels=config.concept_in_channel,
        y_in_channels=config.y_in_channel,
        max_in_len=config.max_in_len,
        max_y_len=config.max_y_len,
        y_null_embedding=torch.zeros(config.max_y_len, config.y_in_channel).to(
            device
        ),  # Hack to avoid null class
        y_null_embedding_mask=torch.ones(config.max_y_len, dtype=torch.bool).to(
            device
        ),  # Hack to avoid null class
    )
    state_dict = find_model(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # important!
    return model

def load_model_unconditional(
    ckpt: str,
    config: Config,
    device: Union[str, int] = "cuda",
    logger: Logger = Logger(name="load_model", level=0),
):
    model = LDiT_models(logger)[config.model](
        in_channels=config.in_channel,
        concept_in_channels=config.concept_in_channel,
        max_in_len=config.max_in_len,
        is_unconditional=True,
    )
    state_dict = find_model(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # important!
    return model

def extract_concepts_from_prompts(
    prompts: List[str],
    model_id="Qwen/Qwen3-8B",
    batch_size=4,
    device="cuda",
):
    

    if "meta-llama" in model_id:
        print(f"Using Llama object count prompt handler with model_id: {model_id}")
        prompt_handler = LlamaObjectCountPromptHandler(
            model_id=model_id,
            batch_size=batch_size,
            device=device,
        )
    elif "gpt" in model_id:
        print(f"Using GPT object count prompt handler with model_id: {model_id}")
        prompt_handler = GPTObjectCountPromptHandler(
            model_id=model_id,
            batch_size=batch_size,
        )
    elif "qwen" in model_id.lower():
        print(f"Using Qwen object count prompt handler with model_id: {model_id}")
        prompt_handler = QwenObjectCountPromptHandler(
            model_id=model_id,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise NotImplementedError("Model not supported")
    info_objects = prompt_handler.get_info(prompts)
    results = []
    for info_object in info_objects:
        result = []
        for obj, count in info_object.objectWithCounts.items():
            try:
                result.extend([obj] * int(count))
            except Exception as e:
                print(f"Error: {e}")
                result.extend([obj])
        results.append(result)

    return results


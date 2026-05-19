import argparse

import torch
from diffusers import StableDiffusionGLIGENPipeline
from sentence_transformers import SentenceTransformer  # type: ignore

from layousyn.config import Config
from layousyn.diffusion import create_diffusion
from layousyn.evaluation.common import (
    extract_concepts_from_prompts,
    load_model,
    sample_with_cfg,
)
from layousyn.model.preprocessor import Preprocessor
from layousyn.model.t5_google import T5EmbedderGoogle
from layout_evaluation import LayoutPlot, LayoutType

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args, config: Config):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = load_model(args.ckpt, config, device=device)
    diffusion = create_diffusion(
        args.num_sampling_steps,
        alpha_scale=config.scale,
        noise_schedule=config.noise_schedule,
        diffusion_steps=config.diffusion_steps,
    )

    # get concept labels
    if args.concepts:
        # get concepts from args
        concept_labels = args.concepts.split(",")
    else:
        # extract with Llama-3.1
        concept_labels = extract_concepts_from_prompts([args.caption])[0]
    print("Concepts: ", concept_labels)

    # get T5 encoder and senetence encoder
    label_encoder = SentenceTransformer(
        f"sentence-transformers/sentence-t5-{config.t5_size}",
        device=device,  # type: ignore
    )
    caption_encoder = T5EmbedderGoogle(
        dir_or_name=f"t5-v1_1-{config.t5_size}",
        device=device,
        model_max_length=config.max_y_len,
    )

    # sample bboxs
    bboxs = sample_with_cfg(
        captions=[args.caption],
        labels_set=[concept_labels],
        config=config,
        diffusion=diffusion,
        model=model,
        label_encoder=label_encoder,
        caption_encoder=caption_encoder,
        cfg_scale=args.cfg_scale,
        ar_int=args.ar,
        device=device,
        return_samples=False,
    )

    # Preprocess the layout
    preprocessor = Preprocessor(config.layout_type).to(device)
    layouts = preprocessor.to_layout(bboxs, [concept_labels])
    layouts = [layout.to(LayoutType.XYXY) for layout in layouts]

    # save layout to scene_layout.png
    height = args.height
    width = int(height * args.ar)
    layout_plotter = LayoutPlot()
    img = layout_plotter.plot_bbox_on_img(
        layouts[0],
        width=width,
        height=height,
        save_path="scene_layout.png",
        add_label_text=True,
    )

    # load GLIGEN pipeline
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    # generate image with GLIGEN
    img = pipe(
        prompt=args.caption,
        gligen_phrases=concept_labels,
        gligen_boxes=layouts[0].bboxs,
        gligen_scheduled_sampling_beta=1,
        output_type="pil",
        num_inference_steps=50,
    ).images[0]
    img.save(f"scene.png")

    # plot bboxs on the image
    img_annotated = layout_plotter.plot_bbox_on_img(
        layouts[0], image=img, save_path=f"scene_annotated.png"
    )
    img_annotated.save(f"scene_annotated.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    # arguments to control the layout
    parser.add_argument(
        "--caption",
        type=str,
        required=True,
        help="Caption for Layout Generation",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        required=False,
        help="Comma separated list of concepts to capture in the layout (Example: 'apple,table')",
    )
    parser.add_argument(
        "--height", type=int, default=256, help="Height of the generated image"
    )
    parser.add_argument(
        "--ar",
        type=float,
        default=1.0,
        help="aspect ratio i.e width/height",
    )

    # arguments to control layout generation quality
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--num-sampling-steps", type=str, default="40")

    # arguments to load layout generation trained model
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a LDiT checkpoint",
    )
    parser.add_argument(
        "--ckpt-config",
        type=str,
        required=True,
        help="Path to a LDiT checkpoint",
    )

    # load arguments
    args = parser.parse_args()
    config = Config.from_json(args.ckpt_config)

    main(args, config)

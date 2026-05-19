"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import timedelta
from time import time
from typing import Optional

import comet_ml
import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm  # drawtoon patch: progress bar

from layousyn.config import Config
from layousyn.data import ConcatDataset, EmbeddedDataset, get_datasets
from layousyn.diffusion import create_diffusion
from layousyn.download import find_model
from layousyn.evaluation.coco_evaluation import run_coco_grounded_evaluation
from layousyn.evaluation.nsr_spatial_evaluation import run_nsr_spatial_evaluation
from layousyn.evaluation.qualitative_evaluation import run_qualitative_evaluation
from layousyn.model.dit import LDiT_models
from layousyn.model.preprocessor import Preprocessor
from layousyn.model.t5_google import T5EmbedderGoogle

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


def collate_fn(batch):
    """
    Collate function for the dataloader.
    """
    (
        bboxs,
        bboxs_padding_mask,
        cats_encoded,
        caption_input_ids,
        caption_attention_mask,
        asp_ratios,
    ) = zip(*batch)
    return (
        torch.stack(bboxs),
        torch.stack(bboxs_padding_mask),
        torch.stack(cats_encoded),
        torch.stack(caption_input_ids),
        torch.stack(caption_attention_mask),
        torch.tensor(asp_ratios),
    )


def get_raw_model(model, is_torch_compile: bool = False, is_ddp: bool = False):
    """
    Get the raw model from a DDP or torch.compile wrapped model.
    """
    if is_ddp:
        model = model.module
    if is_torch_compile:
        model = model._orig_mod
    return model


def main(
    config: Config,
    debug: bool = False,
    is_torch_compile: bool = True,
    resume_ckpt: Optional[str] = None,
    experiment: Optional[comet_ml.Experiment] = None,
):
    """
    Trains a new DiT model.
    """
    assert (
        config.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(
            config.result_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)

        # set to current time to second
        experiment_index = (
            str(time()).replace(".", "-").replace(":", "-").replace(" ", "-")
        )
        model_string_name = config.model.replace(
            "/", "-"
        )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        if debug:
            experiment_dir = (
                f"{config.result_dir}/debug-{model_string_name}-{experiment_index}"
            )
        else:
            experiment_dir = f"{config.result_dir}/{model_string_name}-{experiment_index}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Save config to experiment directory:
        config.to_json(f"{experiment_dir}/config.json")

        # setup tensorboard
        tb_writer = SummaryWriter(log_dir=experiment_dir)

        # log the dataclass config keys to tensorboard
        for key, value in config.to_dict().items():
            tb_writer.add_text(key, str(value))
    else:
        tb_writer = None
        logger = create_logger(None)
    logger.info(config)

    # setup encoders
    label_encoder = SentenceTransformer(
        f"sentence-transformers/sentence-t5-{config.t5_size}",
        device=device,  # type: ignore
    )
    caption_encoder = T5EmbedderGoogle(
        dir_or_name=f"t5-v1_1-{config.t5_size}",
        device=device,
        model_max_length=config.max_y_len,
        torch_dtype=torch.bfloat16,
        is_torch_compile=is_torch_compile,
    )

    # Get datasets and convert to embedded dataset
    datasets = get_datasets(datasets_dict=config.datasets, logger=logger)
    combined_dataset = ConcatDataset(datasets)
    if rank == 0:
        EmbeddedDataset.store_embeddings(
            combined_dataset, config, label_encoder, device=device
        )
    dist.barrier(device_ids=[device])
    embedded_dataset = EmbeddedDataset(
        dataset=combined_dataset,
        embed_dir=config.embed_dir,
        concept_in_channel=config.concept_in_channel,
        caption_encoder=caption_encoder,
        max_bbox=config.max_in_len,
        layout_type=config.layout_type,
    )
    logger.info(f"Embedded dataset contains {len(embedded_dataset):,} layouts")

    # Create model:
    y_null_embedding, y_null_embedding_mask = (
        embedded_dataset.get_null_caption_embedding(caption_encoder)
    )
    model = LDiT_models(logger)[config.model](
        in_channels=config.in_channel,
        concept_in_channels=config.concept_in_channel,
        max_in_len=config.max_in_len,
        y_in_channels=config.y_in_channel,
        max_y_len=config.max_y_len,
        y_null_embedding=y_null_embedding,
        y_null_embedding_mask=y_null_embedding_mask,
    )
    if resume_ckpt:
        logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
        state_dict = find_model(resume_ckpt)
        model.load_state_dict(state_dict)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = torch.compile(model.to(device)) if is_torch_compile else model.to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # setup diffusion for training and validation
    diffusion_train = create_diffusion(
        timestep_respacing="",
        alpha_scale=config.scale,
        noise_schedule=config.noise_schedule,
        diffusion_steps=config.diffusion_steps,
    )  # default: 1000 steps, linear noise schedule
    diffusion_val = create_diffusion(
        timestep_respacing=str(int(config.diffusion_steps / 10)),  # use 10x fewer steps for validation
        alpha_scale=config.scale,
        noise_schedule=config.noise_schedule,
        diffusion_steps=config.diffusion_steps,
    )  # default: 1000 steps, linear noise schedule with 10x fewer steps for validation
    

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    # setup dataloader
    sampler = DistributedSampler(
        embedded_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=config.global_seed,
    )
    loader = DataLoader(
        embedded_dataset,
        batch_size=int(config.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    logger.info(f"Embedded dataset contains {len(embedded_dataset):,} layouts")

    # Prepare models for training:
    update_ema(
        ema,
        get_raw_model(model, is_torch_compile=is_torch_compile, is_ddp=True),
        decay=0,
    )  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Preprocessor
    preprocessor = Preprocessor(config.layout_type).to(device)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss_total = 0
    running_loss_mse = 0
    running_loss_vlb = 0
    running_grad_total_norm = 0
    start_time = time()

    # drawtoon patch: build a parallel val loader if config has val_datasets
    val_loader = None
    val_jsonl_path = getattr(config, "val_jsonl_path", None) or os.environ.get("LAYOUSYN_VAL_JSONL", "")
    if val_jsonl_path and os.path.exists(val_jsonl_path):
        from layousyn.data import MangaLayout
        val_raw = MangaLayout(jsonl_path=val_jsonl_path, logger=logger)
        val_embedded = EmbeddedDataset(
            dataset=val_raw,
            embed_dir=config.embed_dir,  # reuse train's concept cache
            concept_in_channel=config.concept_in_channel,
            caption_encoder=caption_encoder,
            max_bbox=config.max_in_len,
            layout_type=config.layout_type,
        )
        val_batch_per_rank = max(1, int(config.global_batch_size // dist.get_world_size()))
        val_loader = DataLoader(
            val_embedded,
            batch_size=val_batch_per_rank,
            shuffle=False,
            num_workers=max(1, config.num_workers // 2),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(f"drawtoon: val_jsonl_path set, val_loader has {len(val_embedded):,} rows")

    # Training loop:
    logger.info(f"Training for {config.epochs} epochs...")
    scaler = GradScaler()
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        pbar = tqdm(  # drawtoon patch: progress bar over the per-epoch loader
            loader,
            desc=f"epoch {epoch}",
            disable=(rank != 0),
            dynamic_ncols=True,
        )
        for (
            bboxs,
            bboxs_padding_mask,
            cats_encoded,
            caption_input_ids,
            caption_padding_mask,
            asp_ratios,
        ) in pbar:
            bboxs = preprocessor(bboxs.to(device))
            bboxs_padding_mask = bboxs_padding_mask.to(device)
            cats_encoded = cats_encoded.to(device)
            caption_input_ids = caption_input_ids.to(device)
            caption_padding_mask = caption_padding_mask.to(device)
            asp_ratios = asp_ratios.to(device)

            # Get caption embeddings
            y_embed = caption_encoder.get_text_embeddings_from_tokens(
                caption_input_ids, caption_padding_mask
            )

            # Run the model and calculate loss
            t = torch.randint(
                0, diffusion_train.num_timesteps, (bboxs.shape[0],), device=device
            )
            model_kwargs = dict(
                ar=asp_ratios,
                x_enc=cats_encoded,
                y=y_embed,
                y_padding_mask=caption_padding_mask,
            )
            opt.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_dict = diffusion_train.training_losses(
                    model, bboxs, bboxs_padding_mask, t, model_kwargs
                )
                total_loss = loss_dict["loss"].mean()

            # Backward pass and gradient clipping:
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(opt)
            scaler.update()
            update_ema(
                ema,
                get_raw_model(model, is_torch_compile=is_torch_compile, is_ddp=True),
            )

            # Log loss values:
            running_loss_total += total_loss.item()
            running_loss_mse += loss_dict["mse"].mean().item()
            running_loss_vlb += loss_dict["vb"].mean().item()
            running_grad_total_norm += grad_total_norm.item()
            log_steps += 1
            train_steps += 1
            # drawtoon patch: live loss display on the tqdm bar
            if rank == 0:
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", step=train_steps)
            if train_steps % config.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_total_loss = torch.tensor(
                    running_loss_total / log_steps, device=device
                )
                avg_mse_loss = torch.tensor(running_loss_mse / log_steps, device=device)
                avg_vlb_loss = torch.tensor(running_loss_vlb / log_steps, device=device)
                dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_mse_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_vlb_loss, op=dist.ReduceOp.SUM)
                avg_total_loss = avg_total_loss.item() / dist.get_world_size()
                avg_mse_loss = avg_mse_loss.item() / dist.get_world_size()
                avg_vlb_loss = avg_vlb_loss.item() / dist.get_world_size()
                avg_grad_total_norm = running_grad_total_norm / log_steps
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_total_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                # Reset monitoring variables:
                running_loss_total = 0
                running_loss_mse = 0
                running_loss_vlb = 0
                log_steps = 0
                start_time = time()

                # Log to tensorboard:
                if rank == 0:
                    tb_writer.add_scalar("train/loss", avg_total_loss, train_steps)  # type: ignore
                    tb_writer.add_scalar("train/mse", avg_mse_loss, train_steps)  # type: ignore
                    tb_writer.add_scalar("train/vlb", avg_vlb_loss, train_steps)  # type: ignore
                    tb_writer.add_scalar("train/grad_norm", avg_grad_total_norm, train_steps)  # type: ignore

            # Save DiT checkpoint:
            if rank == 0 and train_steps > 0:
                if train_steps % config.ckpt_every == 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"  # type: ignore
                    logger.info(f"Saving checkpoint to {checkpoint_path}")
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": config,
                    }
                    torch.save(checkpoint, checkpoint_path)

                    # drawtoon patch: upstream NSR/COCO/Qualitative eval bypassed.
                    # Those pipelines call Qwen3-8B + DDIM rollouts and take ~10h
                    # per checkpoint. Drawtoon uses its own val-loss pass at the
                    # end of each epoch (see below). Re-enable with
                    # LAYOUSYN_RUN_UPSTREAM_EVAL=1.
                    if os.environ.get("LAYOUSYN_RUN_UPSTREAM_EVAL") == "1":
                        results_dict = run_nsr_spatial_evaluation(
                            config, model=ema, preprocessor=preprocessor,
                            diffusion=diffusion_val, label_encoder=label_encoder,
                            caption_encoder=caption_encoder,
                            val_file="layout_evaluation/nsr/spatial.val.json",
                            output_dir=f"{experiment_dir}/evaluations_{train_steps}/nsr_spatial",
                            cfg_scales=[1.0, 2.0, 4.0, 8.0], device=device,
                        )
                        for result_type, result in results_dict.items():
                            for key, value in result.items():
                                tb_writer.add_scalar(f"val/{result_type}/{key}", value, train_steps)
                        run_qualitative_evaluation(
                            config, model=ema, diffusion=diffusion_val,
                            preprocessor=preprocessor, label_encoder=label_encoder,
                            caption_encoder=caption_encoder,
                            output_dir=f"{experiment_dir}/evaluations_{train_steps}/qualitative",
                            cfg_scales=[1.0, 2.0, 4.0, 8.0], device=device,
                        )
                        results_dict = run_coco_grounded_evaluation(
                            config, model=ema, preprocessor=preprocessor,
                            diffusion=diffusion_val, label_encoder=label_encoder,
                            caption_encoder=caption_encoder,
                            val_file="layout_evaluation/coco_grounded/coco_grounded.val.json",
                            output_dir=f"{experiment_dir}/evaluations_{train_steps}/coco_grounded",
                            partial_file=True, cfg_scales=[2.0, 4.0], device=device,
                        )
                        for result_type, result in results_dict.items():
                            for key, value in result.items():
                                tb_writer.add_scalar(f"val/{result_type}/{key}", value, train_steps)

            dist.barrier(device_ids=[device])

        # drawtoon patch: end-of-epoch val-loss pass on our held-out val.jsonl
        if val_loader is not None:
            model.eval()
            val_running_total = 0.0
            val_running_mse = 0.0
            val_running_vlb = 0.0
            val_batches = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"val epoch {epoch}", disable=(rank != 0), dynamic_ncols=True)
                for (
                    v_bboxs, v_bboxs_padding_mask, v_cats_encoded,
                    v_caption_input_ids, v_caption_padding_mask, v_asp_ratios,
                ) in val_pbar:
                    v_bboxs = preprocessor(v_bboxs.to(device))
                    v_bboxs_padding_mask = v_bboxs_padding_mask.to(device)
                    v_cats_encoded = v_cats_encoded.to(device)
                    v_caption_input_ids = v_caption_input_ids.to(device)
                    v_caption_padding_mask = v_caption_padding_mask.to(device)
                    v_asp_ratios = v_asp_ratios.to(device)
                    v_y_embed = caption_encoder.get_text_embeddings_from_tokens(
                        v_caption_input_ids, v_caption_padding_mask
                    )
                    v_t = torch.randint(
                        0, diffusion_train.num_timesteps, (v_bboxs.shape[0],), device=device
                    )
                    v_kwargs = dict(
                        ar=v_asp_ratios, x_enc=v_cats_encoded,
                        y=v_y_embed, y_padding_mask=v_caption_padding_mask,
                    )
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        v_loss_dict = diffusion_train.training_losses(
                            model, v_bboxs, v_bboxs_padding_mask, v_t, v_kwargs
                        )
                    val_running_total += v_loss_dict["loss"].mean().item()
                    val_running_mse += v_loss_dict["mse"].mean().item()
                    val_running_vlb += v_loss_dict["vb"].mean().item()
                    val_batches += 1
            denom = max(1, val_batches)
            val_loss = val_running_total / denom
            val_mse = val_running_mse / denom
            val_vlb = val_running_vlb / denom
            # reduce across ranks
            val_loss_t = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
            val_loss_avg = val_loss_t.item() / dist.get_world_size()
            logger.info(
                f"[epoch {epoch}] VAL loss={val_loss_avg:.4f}  mse={val_mse:.4f}  vb={val_vlb:.4f}  ({val_batches} batches)"
            )
            if rank == 0 and tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss_avg, train_steps)
                tb_writer.add_scalar("val/mse", val_mse, train_steps)
                tb_writer.add_scalar("val/vlb", val_vlb, train_steps)
            model.train()

    # Save final checkpoint:
    if dist.get_rank() == 0:
        # save final checkpoint
        checkpoint_path = f"{checkpoint_dir}/final.pt"  # type: ignore
        logger.info(f"Saving final checkpoint to {checkpoint_path}")
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": config,
        }
        torch.save(checkpoint, checkpoint_path)

        # save the final model to comet if enabled
        if experiment:
            experiment.log_model("final_model", checkpoint_path)
    dist.barrier(device_ids=[device])

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Arguments for saving/loading data
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file. Provided arguments will override config file.",
    )

    # argument to override config file
    parser.add_argument(
        "--embed-dir",
        type=str,
        default="",  # set it to a fast I/O storage
        help="Path to save concept and caption embeddings. Note: This will override the embed_dir in the config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="The model to train. If not provided, the model in the config file will be used.",
    )

    # arguments for resuming training
    parser.add_argument(
        "--result-dir",
        type=str,
        default="",
        help="Path to save results. Note: This will override the result_dir in the config file.",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    # argument to enable comet
    parser.add_argument(
        "--use-comet",
        action="store_true",
        help="Use comet for logging. Set `COMET_API_KEY` in your environment variables.",
    )

    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode. If enabled, will run with torch.compile disabled.",
    )

    # parse arguments
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl", timeout=timedelta(minutes=120))

    # Load config file and override with provided arguments
    config = Config.from_json(args.config)

    # check if comet is enabled
    experiment = None
    if args.use_comet and dist.get_rank() == 0:
        experiment = comet_ml.Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name="layoutdit",
        )

        # log the config file
        experiment.log_parameters(config.to_dict())

    if args.embed_dir:
        config.embed_dir = args.embed_dir
    if args.model:
        config.model = args.model
    if args.result_dir:
        config.result_dir = args.result_dir

    main(
        config,
        resume_ckpt=args.resume_ckpt,
        experiment=experiment,
        is_torch_compile=not args.debug,
        debug=args.debug,
    )

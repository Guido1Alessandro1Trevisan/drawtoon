import os
from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        ddp_options = {
            "find_unused_parameters": _env_bool(
                "AITK_DDP_FIND_UNUSED_PARAMETERS",
                True,
            ),
        }
        if _env_bool("AITK_DDP_STATIC_GRAPH", False):
            ddp_options["static_graph"] = True
        try:
            ddp_kwargs = DistributedDataParallelKwargs(**ddp_options)
        except TypeError:
            ddp_options.pop("static_graph", None)
            ddp_kwargs = DistributedDataParallelKwargs(**ddp_options)
        # PyTorch default NCCL pg timeout is 10 min — too tight when one rank
        # is stalled on a cold S3-mount page fetch. Bump to 30 min (env-tunable).
        pg_timeout_seconds = _env_int("AITK_NCCL_TIMEOUT_SECONDS", 1800)
        init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=pg_timeout_seconds))
        global_accelerator = Accelerator(
            gradient_accumulation_steps=1,
            kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
        )
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model

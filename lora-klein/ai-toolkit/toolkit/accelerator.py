import os

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        global_accelerator = Accelerator(
            gradient_accumulation_steps=1,
            kwargs_handlers=[ddp_kwargs],
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

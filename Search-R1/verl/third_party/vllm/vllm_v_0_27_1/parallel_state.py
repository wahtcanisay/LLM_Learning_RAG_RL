"""Single-node compatibility surface used by Search-R1's sharding manager."""

import torch.distributed as dist


def initialize_parallel_state(tensor_model_parallel_size=1, **_kwargs):
    if tensor_model_parallel_size != 1:
        raise NotImplementedError(
            "The vLLM 0.27.1 compatibility adapter currently supports TP=1 only."
        )


def get_tensor_model_parallel_world_size():
    return 1


def get_tensor_model_parallel_group():
    return dist.group.WORLD if dist.is_initialized() else None


def get_tensor_model_parallel_src_rank():
    return 0


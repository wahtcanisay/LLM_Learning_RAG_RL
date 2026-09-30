"""Legacy veRL rollout facade backed by vLLM 0.27.1 public APIs.

The Search-R1 trainer expects a synchronous engine that accepts token-id
prompts, returns padded token/log-prob tensors, can release GPU memory, and can
receive the current FSDP actor weights.  vLLM 0.27.1 exposes those capabilities
through a different API; this module performs only that translation.
"""

import os
import random
from collections.abc import Iterator, Mapping

import numpy as np
import torch

# vLLM 0.27 defaults to its V2 GPU runner, whose staged host buffers require
# CUDA UVA.  Docker Desktop/WSL2 reports UVA unavailable on this RTX 5090 even
# though ordinary CUDA tensors work.  The supported V1 runner does not impose
# that host-memory requirement and retains the same public LLM API.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
# The FlashInfer wheel installed by vLLM mis-detects SM120 under WSL2 and
# rejects the 5090 during its JIT sampler build.  vLLM's native sampler is the
# supported fallback and has the same sampling semantics for this rollout.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Keep EngineCore in this process on the single-GPU WSL2 setup.  Cross-process
# CUDA IPC is unavailable and NCCL rejects two ranks bound to the same GPU.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM as _VLLM


def _reload_model_weights(worker, weights):
    """Load HF-format tensors into the in-process vLLM worker model."""
    from vllm.model_executor.model_loader.mtp_validation import (
        disable_mtp_completeness_check,
    )
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
    )

    gpu_worker = getattr(worker, "worker", worker)
    model = gpu_worker.model_runner.model
    initialize_layerwise_reload(model)
    with disable_mtp_completeness_check():
        model.load_weights(weights)
    finalize_layerwise_reload(model, gpu_worker.vllm_config.model_config)


class _MappingWeightSource:
    """Expose a state-dict mapping as vLLM's re-iterable WeightSource."""

    def __init__(self, weights: Mapping[str, torch.Tensor]):
        self._weights = weights

    def metadata(self):
        from vllm.distributed.weight_transfer.base import ParamMeta

        return [
            ParamMeta(name=name, dtype=tensor.dtype, shape=tuple(tensor.shape))
            for name, tensor in self._weights.items()
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        from vllm.distributed.weight_transfer.base import materialize_full_tensor

        for name, tensor in self._weights.items():
            yield name, materialize_full_tensor(tensor).detach().contiguous()


class LLM:
    """Compatibility class preserving Search-R1's legacy LLM method names."""

    def __init__(
        self,
        actor_module,
        *,
        tokenizer,
        model_hf_config,
        tensor_parallel_size=1,
        dtype="auto",
        enforce_eager=False,
        gpu_memory_utilization=0.5,
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        kv_cache_memory_bytes=None,
        **_legacy_kwargs,
    ):
        if tensor_parallel_size != 1:
            raise NotImplementedError(
                "The vLLM 0.27.1 Search-R1 adapter currently supports TP=1 only."
            )

        model_path = getattr(model_hf_config, "_name_or_path", None)
        if not model_path:
            model_path = getattr(tokenizer, "name_or_path", None)
        if not model_path:
            raise ValueError("Cannot determine the Hugging Face model path for vLLM.")

        tokenizer_path = getattr(tokenizer, "name_or_path", None) or model_path
        self.pad_token_id = tokenizer.pad_token_id
        self._sleeping = False
        # In-process vLLM seeds Python, NumPy and Torch globally. Preserve the
        # trainer RNG streams so constructing rollout does not change actor
        # minibatch order, dropout or later sampling state.
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        try:
            self._engine = _VLLM(
                model=str(model_path),
                tokenizer=str(tokenizer_path),
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                kv_cache_memory_bytes=kv_cache_memory_bytes,
                enable_sleep_mode=True,
            )
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_states)

    def generate(
        self,
        prompts=None,
        sampling_params=None,
        prompt_token_ids=None,
        use_tqdm=False,
        **kwargs,
    ):
        token_prompts = prompt_token_ids if prompt_token_ids is not None else prompts
        outputs = self._engine.generate(
            token_prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            **kwargs,
        )

        token_rows = []
        logprob_rows = []
        for request_output in outputs:
            for completion in request_output.outputs:
                token_ids = list(completion.token_ids)
                token_rows.append(token_ids)
                sampled_logprobs = []
                for token_id, candidates in zip(token_ids, completion.logprobs or []):
                    candidate = candidates.get(token_id)
                    sampled_logprobs.append(
                        float(candidate.logprob) if candidate is not None else 0.0
                    )
                sampled_logprobs.extend([0.0] * (len(token_ids) - len(sampled_logprobs)))
                logprob_rows.append(sampled_logprobs)

        max_length = max((len(row) for row in token_rows), default=0)
        padded_tokens = [
            row + [self.pad_token_id] * (max_length - len(row)) for row in token_rows
        ]
        padded_logprobs = [
            row + [0.0] * (max_length - len(row)) for row in logprob_rows
        ]
        return (
            torch.tensor(padded_tokens, dtype=torch.long),
            torch.tensor(padded_logprobs, dtype=torch.float32),
        )

    def sync_model_weights(self, params, load_format=None):
        del load_format  # The model consumes HF-format names regardless of source sharding.
        if self._sleeping:
            # Restore only the destination weight buffers while the KV cache is
            # still released. This avoids the RLHF peak where actor weights,
            # vLLM weights and KV cache are all resident at once.
            self._engine.wake_up(tags=["weights"])

        source = _MappingWeightSource(params)
        self._engine.collective_rpc(
            _reload_model_weights,
            args=(list(source),),
        )
        self._engine.reset_prefix_cache()
        if self._sleeping:
            self._engine.wake_up(tags=["kv_cache"])
            self._sleeping = False

    def offload_model_weights(self):
        if not self._sleeping:
            # Search-R1 calls this only after synchronous generation returns,
            # so there are no in-flight requests to preserve.  In-process
            # EngineCore does not support vLLM's multi-process-only wait mode.
            # Level 2 discards stale rollout weights as well as KV cache. The
            # next sync supplies the current actor weights explicitly.
            self._engine.sleep(level=2, mode="abort")
            self._sleeping = True

    def init_cache_engine(self):
        if self._sleeping:
            self._engine.wake_up()
            self._sleeping = False

    def free_cache_engine(self):
        self._engine.reset_prefix_cache()


# The old vendored package exported both names.  Search-R1 only constructs LLM,
# but retaining the alias keeps imports stable for the local veRL code.
LLMEngine = LLM

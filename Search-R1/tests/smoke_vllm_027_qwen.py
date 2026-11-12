"""Real-GPU smoke test for the vLLM 0.27.1 compatibility adapter."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from verl.third_party.vllm import LLM


MODEL_PATH = Path(
    "/workspace/code_list/some tricks/LLMLeanring/models/Qwen2.5-3B-Instruct"
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    engine = LLM(
        None,
        tokenizer=tokenizer,
        model_hf_config=model_config,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        max_model_len=256,
    )
    prompt_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    token_ids, logprobs = engine.generate(
        prompt_token_ids=[prompt_ids],
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=8,
            logprobs=1,
        ),
        use_tqdm=False,
    )
    print("TOKEN_SHAPE", tuple(token_ids.shape))
    print("LOGPROB_SHAPE", tuple(logprobs.shape))
    print("TEXT", tokenizer.decode(token_ids[0].tolist()))

    # Match the hybrid actor/rollout lifecycle: release vLLM memory, materialize
    # the current actor, load its state dict into the in-process vLLM worker,
    # then generate again through the unchanged legacy facade.
    engine.offload_model_weights()
    actor = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        actor.model.norm.weight.zero_()
    engine.sync_model_weights(actor.state_dict())
    synced_token_ids, synced_logprobs = engine.generate(
        prompt_token_ids=[prompt_ids],
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=4,
            logprobs=1,
        ),
        use_tqdm=False,
    )
    print("SYNC_TOKEN_SHAPE", tuple(synced_token_ids.shape))
    print("SYNC_LOGPROB_SHAPE", tuple(synced_logprobs.shape))
    print("SYNC_TEXT", tokenizer.decode(synced_token_ids[0].tolist()))
    sync_changed_output = not torch.equal(token_ids[:, :4], synced_token_ids)
    print("SYNC_CHANGED_OUTPUT", sync_changed_output)
    if not sync_changed_output:
        raise AssertionError("vLLM output did not change after actor weight sync")
    engine.offload_model_weights()
    print("SMOKE_OK")


if __name__ == "__main__":
    main()

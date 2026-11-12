"""单卡师生资源切换、失败轨迹提取、教师评分与分数回填。"""

import torch
from MedSearch.distillation.teacher import score_tokens
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

def prepare_teacher_phase(runtime) -> None:
    """教师评分前卸载学生 actor 与 optimizer。

    Args:
        runtime: 提供 config 和 actor_rollout_wg 的训练运行对象。
            调用方应先完成 rollout 休眠和 reward 计算。

    Returns:
        None；通过 worker group 将学生状态卸载到 CPU。
    """
    # 已启用参数自动卸载时，不重复迁移模型参数。
    param_offload = runtime.config.actor_rollout_ref.actor.fsdp_config.param_offload
    runtime.actor_rollout_wg.to(
        "cpu",
        model=not param_offload,
        optimizer=True,
        grad=False,
    )
        


def restore_student_phase(runtime) -> None:
    """教师进程退出后恢复学生 actor 与 optimizer。

    Args:
        runtime: 与卸载时相同的训练运行对象；调用方应在 finally 中恢复。

    Returns:
        None；通过 worker group 将学生状态恢复到训练设备。
    """
    # 模型参数是否显式恢复，与卸载时采用相同配置。
    param_offload = runtime.config.actor_rollout_ref.actor.fsdp_config.param_offload
    runtime.actor_rollout_wg.to(
        "device",
        model=not param_offload,
        optimizer=True,
        grad=False,
    )


def build_teacher_score_jobs(
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
) -> list[dict[str, list[int] | int]]:
    """把连续的学生动作拆成教师评分任务。

    Args:
        prompt_ids: 初始上下文的 token ID。
        response_ids: 学生动作与环境 observation 交错的 token ID。
        response_mask: 与 response_ids 对齐；1 表示学生生成，0 表示环境内容。

    Returns:
        [{
            "prefix_ids": 本段动作前的全部可见 token ID，含历史 observation。
            "target_ids": 本段连续学生生成的 token ID。
            "response_start": 本段在原 response 中的起始索引。
        }]
        没有学生生成位置时返回空列表。
    """

    jobs = []
    start = None # 当前连续学生动作起点，为None表示未进入动作段

    for i, mask in enumerate(response_mask):
        if mask == 1 and start is None:
            # 进入动作段, 标记开始
            start = i 
        elif mask == 0 and start is not None:
            # [start:i] 是刚结束的学生动作

            # observation 保留在后续动作的上下文中，但不作为评分目标。
            prefix_ids = prompt_ids + response_ids[:start]
            target_ids = response_ids[start: i]
            response_start = start
            jobs.append({
                "prefix_ids" : prefix_ids,
                "target_ids" : target_ids,
                "response_start" : response_start,
            })
            start = None

    # 轨迹结束时仍处于动作段，将末段补入任务列表。

    if start is not None:
        prefix_ids = prompt_ids + response_ids[:start]
        target_ids = response_ids[start:]
        response_start = start
        jobs.append({
                        "prefix_ids" : prefix_ids,
                        "target_ids" : target_ids,
                        "response_start" : response_start,
                    })

    return jobs


def score_failed_trajectory(
    teacher_model: torch.nn.Module | None,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    failed: bool,
) -> dict[str, torch.Tensor]:
    """为失败轨迹计算教师分数，并回填到原 response 位置。

    Args:
        teacher_model: 已加载且冻结的教师；成功轨迹允许传 None。
        prompt_ids: 初始上下文 token ID。
        response_ids: 学生动作与检索 observation 交错的 token ID。
        response_mask: 与 response_ids 对齐；学生 token 为 1，环境 token 为 0。
        failed: verifier 是否判定该轨迹答错。

    Returns:
        {
            "teacher_logprobs": 与 response 对齐的教师逐 token logprob。
            "opd_mask": 失败轨迹学生 token 为 1，其他位置为 0。
        }
        两个张量均为长度 len(response_ids) 的 CPU float32 一维张量。
        成功轨迹不调用教师，两个张量均为零。
    """

    # 按完整 response 预留评分位置，未评分位置保持零。
    N = len(response_ids)
    teacher_logprob, opd_mask = torch.zeros(N), torch.zeros(N)

    if failed == False:
        # 当前轨迹答对时跳过教师，OPD 不参与训练。
        return{
            "teacher_logprobs" : teacher_logprob,
            "opd_mask" : opd_mask,
            }
        

    # 按连续动作段拆分，避免将环境 observation 当作教师评分目标。
    jobs = build_teacher_score_jobs(
        prompt_ids,
        response_ids,
        response_mask,
    )

    if not jobs:
        # 没有学生 token 时无需调用教师。
        return{
               "teacher_logprobs" : teacher_logprob,
               "opd_mask" : opd_mask,
            }

    if jobs is not None and teacher_model is None:
        # jobs 非空但是教师模型为空, 报错
        raise ValueError(
            "teacher_model is required to score a failed trajectory with student tokens; "
            "load the teacher before calling score_failed_trajectory."
        )

    
    # 输入放到教师所在设备，每次只评分一个动作段。
    device = next(teacher_model.parameters()).device
    for job in jobs:
        # 添加 batch 维度，构造 [1, P] 上下文和 [1, T] 目标。
        prefix = torch.tensor(job["prefix_ids"],
                              dtype=torch.long,
                              device=device).unsqueeze(0)

        target = torch.tensor(job["target_ids"],
                              dtype=torch.long,
                              device=device,).unsqueeze(0) # unsqueeze适配[B, T]维度

        # 只读取学生实际采样 token 的教师 logprob，不让教师生成新答案。
        scores = score_tokens(
            teacher_model,
            prefix,
            target,
            mode="sampled",
        )
        # 去掉 batch 维度并保存为不参与反传的 CPU float32 分数。
        segment_scores = scores.squeeze(0).detach().cpu().float()

        # 标出teacher logp的位置
        start = job["response_start"]
        end = start + len(job["target_ids"])

        # 按原 response 位置回填，本段学生 token 才启用 OPD。
        teacher_logprob[start:end] = segment_scores
        opd_mask[start:end] = 1


    return {
        "teacher_logprobs" : teacher_logprob,
        "opd_mask" : opd_mask,
    }



        
def build_failure_batch(batch_meta, data) -> list[dict]:
    """从完整已评分批次中提取失败轨迹，保留原始身份与索引。
    
    Args:
        batch_meta: veRL V1 的 KVBatchMeta，保存轨迹 keys 和 tags。
            keys 为轨迹唯一身份；tags 中 is_padding 标记合成补位样本。
        data: 按 keys 顺序读取的完整 TensorDict，包含 uid、prompts、
            responses、response_mask 和 extra_fields。
              
    Returns:
        [{
            "original_index": 当前完整批次中的轨迹索引。
            "group_id": 同题 rollout 共用的 uid。
            "trajectory_id": batch_meta.keys 中的轨迹唯一身份。
            "prompt_ids": 初始上下文 token ID 列表。
            "response_ids": 学生动作与环境内容交错的 token ID 列表。
            "response_mask": 学生 token 为 1、环境内容为 0 的列表。
        }]
        仅提取 acc=0 的非 padding 项；不修改原批次或重新组成 GRPO 组。
    """
    failures = []
    
    for i, tag in enumerate(batch_meta.tags):

        # 合成补位项没有有效判分，读取 acc 前先跳过。
        if tag.get("is_padding", False):
            continue 

        trajectory_id = batch_meta.keys[i]

        row = data[i]
        # 失败依据是 verifier 的 acc，而不是 reward 是否等于零。
        acc = row["extra_fields"]["reward_extra_info"]["acc"]

        if acc is None:
            raise ValueError(
                f"Trajectory {trajectory_id!r} at batch index {i} has no acc value "
                "in extra_fields.reward_extra_info; compute reward before building the failure batch."
            )

        if acc == 1:

            continue

        elif acc == 0:
            # 只带教师评分所需 token 与身份，不把 gold 送给教师。
            failures.append({
                "original_index" : i,
                "group_id": row["uid"],
                "trajectory_id": trajectory_id,
                "prompt_ids": row["prompts"].tolist(),
                "response_ids": row["responses"].tolist(),
                "response_mask": row["response_mask"].tolist(),
            })


    return failures

def score_failed_batch(
        teacher_model: torch.nn.Module | None,
        failure_batch: list[dict],
) -> list[dict]:
    """逐条评分失败轨迹，保留身份供后续回填。
    
    Args: 
        teacher_model: 已加载且冻结的教师；空失败列表允许传 None。
        failure_batch: build_failure_batch 返回的列表，每项包含：
            {
                "original_index": 原完整批次索引。
                "group_id": 同题 rollout 共用的 uid。
                "trajectory_id": 轨迹唯一身份。
                "prompt_ids": 初始上下文 token ID 列表。
                "response_ids": 完整 response token ID 列表。
                "response_mask": 学生 token 为 1、环境内容为 0 的列表。
            }

    Returns:
        [{
            "original_index": 原完整批次索引，不按失败子集重新编号。
            "trajectory_id": 轨迹唯一身份。
            "teacher_logprobs": 与该条 response 对齐的 CPU float32 分数。
            "opd_mask": 与该条 response 对齐的 CPU float32 OPD 位置标记。
        }]
        空失败列表返回 []；评分异常向上传递，不返回部分成功结果。
    """

    results = []

    # 串行复用已加载的教师，避免为每条轨迹重新加载模型。
    for item in failure_batch:

        prompt_ids = item["prompt_ids"]
        response_ids = item["response_ids"]
        response_mask = item["response_mask"]

        # 输入已经是失败子集，每条均按 failed=True 评分。
        scores = score_failed_trajectory(
            teacher_model=teacher_model,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            failed=True,
        )

        # 保留完整批次索引与身份，供后续散射回填。
        results.append({
            "original_index": item["original_index"],
            "trajectory_id": item["trajectory_id"],
            "teacher_logprobs": scores["teacher_logprobs"],
            "opd_mask": scores["opd_mask"],
        })

    return results


def attach_teacher_scores(batch_meta, data, scores: list[dict]):
    """将失败轨迹的教师分数按原始索引回填到完整批次。

    Args:
        batch_meta: veRL V1 的 KVBatchMeta；keys 与 data 的轨迹顺序一致。
        data: 完整批次 TensorDict；responses 保存每条轨迹的实际 token 序列。
        scores: score_failed_batch 返回的列表，允许顺序打乱或为空：
            [{
                "original_index": 要回填的原完整批次索引。
                "trajectory_id": 用于核对 batch_meta.keys 的轨迹身份。
                "teacher_logprobs": 与该条 response 等长的 CPU float32 一维分数。
                "opd_mask": 与该条 response 等长的 CPU float32 一维位置标记。
            }]

    Returns:
        添加以下字段后的原 data：
        {
            "teacher_logprobs": 按完整批次顺序组织的教师逐 token 分数。
            "opd_mask": 按完整批次顺序组织的 OPD 有效位置标记。
        }
        两个字段逐条与 responses 对齐；未回填轨迹保持全零。
        原有字段、批次顺序和 batch_meta 保持不变。
    """
    # 列表下标对应完整批次索引，成功项与 padding 默认保持零。
    teacher_logprobs_list = []
    opd_mask_list = []

    # 轨迹长度可能不同，按各自 response 长度分配一维张量。
    for i in range(len(data)):
        response_length = len(data[i]["responses"])

        teacher_logprobs = torch.zeros(
            response_length,
            dtype=torch.float32,
            device="cpu"
        )

        opd_mask = torch.zeros(
            response_length,
            dtype=torch.float32,
            device="cpu",
        )

        teacher_logprobs_list.append(teacher_logprobs)
        opd_mask_list.append(opd_mask)


    for score in scores:
        # 使用原始索引定位，不依赖 scores 的排列顺序。
        index = score["original_index"]

        # 评分回填校验
        # region
        if not (0 <= index < len(data)):
            raise ValueError(
                f"Score index {index} is out of range for batch size {len(data)}."
            )
        if not score["trajectory_id"] == batch_meta.keys[index]:
            raise ValueError(
                f"Trajectory mismatch at batch index {index}: "
                f"expected {batch_meta.keys[index]!r}, got {score['trajectory_id']!r}."
            )

        expected_shape = (len(data[index]["responses"]),)
        for field in ("teacher_logprobs", "opd_mask"):
            if score[field].shape != expected_shape:
                raise ValueError(
                    f"Trajectory {score['trajectory_id']!r}: {field} "
                    f"has shape {tuple(score[field].shape)}, expected {expected_shape}."
                )
        # endregion

        # 替换本地占位张量；全部校验通过前不写入原 data。
        teacher_logprobs_list[index] = score["teacher_logprobs"]
        opd_mask_list[index] = score["opd_mask"]

    # 复用 veRL 组装等长或变长批次张量，再加入训练所需评分字段。
    data.update(list_of_dict_to_tensordict([
        {
            "teacher_logprobs": logprobs,
            "opd_mask": mask,
        }
        for logprobs, mask in zip(teacher_logprobs_list, opd_mask_list)
    ]))   
    
    return data


def run_failure_teacher_phase(runtime, batch_meta, data, score_fn):
    """调度失败轨迹评分, 完成后恢复学生.

    Args:
        runtime: 提供 config 和 actor_rollout_wg 的训练对象.
        batch_meta: 保存轨迹 keys 和 padding tags 的批次信息.
        data: 已判分的完整 TensorDict.
        score_fn: 接收失败轨迹列表, 负责加载教师模型, 调用score_failed_batch, 传回logp, 退出教师进程

    Returns:
        原 data, 新增 teacher_logprobs 和 opd_mask.
    """
    # 提取失败项数据, data仍然保留完整批次
    failures = build_failure_batch(batch_meta=batch_meta, data=data)
    scores = []

    if failures:
        # 卸载学生
        prepare_teacher_phase(runtime)

        # 调用评分, 随后恢复学生
        try:
            scores = score_fn(failures)

        finally:
            restore_student_phase(runtime)

    return attach_teacher_scores(batch_meta, data, scores)

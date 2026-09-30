"""单卡师生资源切换。"""


def prepare_teacher_phase(runtime) -> None:
    """教师评分前卸载学生 actor

    runtime：当前 RayPPOTrainer。调用时 rollout 已休眠、reward 已算完。
    """
    param_offload = runtime.config.actor_rollout_ref.actor.fsdp_config.param_offload
    runtime.actor_rollout_wg.to(
        "cpu",
        model=not param_offload,
        optimizer=True,
        grad=False,
    )
        


def restore_student_phase(runtime) -> None:
    """教师进程退出后恢复学生 actor，供下一次前向使用。

    runtime：与卸载时相同的 RayPPOTrainer。调用方应放在 finally 中。
    """
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
        list[
            dict{
                "prefix_ids": 动作前全部可见token,
                "target_ids": 本段学生生成token,
                "response_start": 动作回复的起始下标,
            }
        ]
    """

    jobs = []
    start = None # 当前连续学生动作起点，为None表示未进入动作段

    for i, mask in enumerate(response_mask):
        if mask == 1 and start is None:
            # 进入动作段, 标记开始
            start = i 
        elif mask == 0 and start is not None:
            # [start:i] 是刚结束的学生动作

            # 记录当前轮动作
            prefix_ids = prompt_ids + response_ids[:start]
            target_ids = response_ids[start: i]
            response_start = start
            jobs.append({
                "prefix_ids" : prefix_ids,
                "target_ids" : target_ids,
                "response_start" : response_start,
            })
            start = None

    # 结束循环后若start不为None则需要将末尾一并记录

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

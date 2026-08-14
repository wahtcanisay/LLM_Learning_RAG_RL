# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""开放域 QA 的规则式 Exact Match reward（学习重要性 P0）。

本文件不加载 reward model。它从完整 ``prompt + rollout`` 中取最终 ``<answer>``，
做 SQuAD 风格规范化，然后给每条轨迹 0/1 分。训练器会把这个序列级分数放到
response 的最后一个有效 token，再由 GRPO 在同题多条 rollout 间计算相对优势。
"""

import re
import string
import random

def normalize_answer(s):
    """小写、去英文冠词/标点并压缩空白，降低表面格式对 EM 的影响。

    这是英文开放域 QA 规则；迁移医学选择题时应改为显式解析 A/B/C/D，不能假设
    这套规范化自动适合 MedQA。
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """预测必须与任一 gold alias 规范化后完全相等。"""
    # 本质还是字符串匹配
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    """宽松版：任一 gold 是预测的子串即可；主训练入口默认不用它。"""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """从完整序列中解析模型最终的 ``<answer>`` 内容。

    输入：
        ``solution_str``（``str``）：tokenizer 解码后的完整
        ``prompt + response``，而不是仅模型 response。字符串可能包含 prompt 中的
        ``<answer> Beijing </answer>`` 格式示例、若干 search/information 轮次和模型
        最终答案。

    输出：
        ``str | None``。至少找到两个完整 ``<answer>...</answer>`` 块时，返回最后
        一个块去除首尾空白后的文本；只有 0 或 1 个块时返回 ``None``。

    调用方式：
        由本文件 ``compute_score_em()`` 和 ``compute_score_subem()`` 调用。
        主训练链使用前者：``RewardManager.__call__()`` 先选择
        ``compute_score_em``，该函数再调用这里提取答案。

    注意官方实现要求至少出现两个 answer 块。Search-R1 的 prompt 自带一个格式示例，
    模型最终输出构成第二个；如果迁移数据时移除了 prompt 示例，这里会返回 ``None``，
    即使模型只输出了一个正确 answer 也得到 0 分。
    """
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # 0/1 个标签都视为格式不完整；这个行为与常见“至少一个”解析器不同。
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """解析最终答案并计算规则式 Exact Match outcome reward。

    输入：
        ``solution_str``（``str``）：完整 ``prompt + response`` 文本，由
        ``RewardManager.__call__()`` 拼接有效 token 后解码得到。
        ``ground_truth``（``dict``）：至少包含 ``{'target': str | list[str]}``；
        ``method``（``str``）是未参与当前分支的兼容参数；``format_score`` 和
        ``score``（``float``）分别是格式正确但答案错误、答案正确时的返回分数。

    输出：
        ``int | float`` 标量。无法解析答案返回 ``0``；规范化后命中任一 gold alias
        返回 ``score``（默认 ``1.0``）；格式可解析但答案错误返回 ``format_score``
        （默认 ``0.0``）。这里不创建 token 维度，token-level 放置由
        ``RewardManager`` 完成。

    调用方式：
        ``main_ppo._select_rm_score_fn(data_source)`` 返回本函数，随后
        ``RewardManager.__call__()`` 以关键字参数调用。函数内部先调用
        ``extract_solution()``，再调用 ``em_check()``；结果最终写到每条 response
        的最后一个有效 token，并由 trainer 传入 GRPO advantage 计算。
    """
    answer = extract_solution(solution_str=solution_str)
    # 约 1/64 采样打印轨迹用于人工检查；random 不参与 reward 数值。
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            # 默认 format_score=0：格式正确但答案错误仍为 0。
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

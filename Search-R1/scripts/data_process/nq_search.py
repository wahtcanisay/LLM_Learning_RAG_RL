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
"""把 Natural Questions (NQ) 转成 Search-R1 所需 Parquet（学习重要性 P0）。

这个脚本不做 tokenization，只定义“题目如何进入 RL 系统”的数据协议：prompt 教模型
使用 think/search/information/answer 标签；gold aliases 放入 ``reward_model``；稳定题号
放入 ``extra_info.index``，供后续同题多 rollout 的 GRPO 分组。
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    """构造 Base 模型的文本协议 prompt。

    prompt 内的 ``<answer> Beijing </answer>`` 是格式示例，也会被最终 reward parser
    看见；模型输出最终 answer 后，完整序列恰好至少包含两个 answer 块。
    """
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    # argparse 是 Python 标准库 CLI 解析器；local_dir 默认输出 train/test.parquet。
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'nq'

    # Hugging Face datasets 负责下载、缓存、map 变换及写 Parquet。
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # split 被闭包捕获，使同一个处理函数可分别标记 train/test。
    def make_map_fn(split):

        def process_fn(example, idx):
            """将一条原始 NQ 样本映射为 veRL 能读取的一行。"""
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            # golden_answers 可能包含多个等价字符串，EM 命中任意一个即得 1 分。
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                # 保持 chat message 结构；RLHFDataset 再决定是否应用 chat template。
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                # style=rule 表示不调用 learned reward model，而用 qa_em.py 打分。
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                # index 会在 dataset loader 中提升为顶层 index，再复制成 GRPO uid。
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    # with_indices=True 把稳定行号 idx 传给 process_fn。
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # HDFS 是 veRL 集群场景的可选路径；本地单机学习可保持 None。
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

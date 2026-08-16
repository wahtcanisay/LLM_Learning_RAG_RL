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

"""把 Search-R1 的 Parquet 样本转换为 veRL rollout batch（学习重要性 P0）。

一行数据既有送进模型的 ``prompt``，也有不参与 tokenization 的元数据，例如：

- ``data_source``：决定调用哪一种规则 reward；
- ``reward_model.ground_truth``：最终答案及别名；
- ``extra_info.index``：同一道题多条 rollout 的分组标识。

``DataLoader`` 通过 ``collate_fn`` 将 tensor 字段 stack，将嵌套 Python 对象保存为
``dtype=object`` 的 NumPy 数组，之后由 veRL 的 ``DataProto`` 统一携带。
"""

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    """合并样本，同时保留 tensor 与任意 Python 元数据。

    输入：
        ``data_list``（``list[dict]``）：同一 mini-batch 的单样本字典列表；每个字典
        来自 ``RLHFDataset.__getitem__()``。其中 token 字段是 ``torch.Tensor``，
        ``data_source/reward_model/index`` 等字段仍是 Python 对象。

    输出：
        ``dict``：

        - tensor 字段通过 ``torch.stack(..., dim=0)`` 变为
          ``[batch_size, seq_len]``；
        - 非 tensor 字段变为 ``np.ndarray(dtype=object)``，长度为 ``batch_size``，
          保持与 tensor 第 0 维逐样本对齐。

    调用方式：
        ``RayPPOTrainer._create_dataloader()`` 创建训练和验证 ``DataLoader`` 时将本函数
        作为 ``collate_fn`` 传入；PyTorch DataLoader 每收集一批 ``__getitem__``
        返回值后自动调用它。合并结果随后由 ``DataProto.from_single_dict()`` 拆分为
        tensor batch 与 non-tensor batch。

    PyTorch 默认 collate 不适合 ``reward_model`` 这类嵌套 dict；这里显式分流：
    tensor 变成形状 ``[batch, seq_len]``，其余对象保持逐样本对齐。
    """
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    从一个或多个 Parquet 文件懒加载单条 RL prompt。

    ``OmegaConf.ListConfig`` 是 Hydra YAML 列表的运行时类型，因此既接受普通 Python
    list，也接受配置文件中的列表。``truncation='error'`` 会让过长 prompt 直接失败，
    便于发现长度配置错误，而不是静默丢掉题目开头。
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        """记录数据配置，准备本地 Parquet，并建立行级 DataFrame。

        输入：
            ``parquet_files``（``str | list[str] | ListConfig``）：一个或多个本地/HDFS
            Parquet 路径；``tokenizer``（``PreTrainedTokenizer``）：负责 chat template、
            tokenization 和 pad token；其余参数控制 prompt 列名、最大长度、缓存、是否
            返回 raw chat 以及过长 prompt 的处理方式。

        输出：
            构造完成的 ``RLHFDataset`` 实例；无显式返回值。实例持有下载后的文件列表
            和拼接后的 ``self.dataframe``，但此时尚未逐行 tokenization。

        调用方式：
            ``RayPPOTrainer._create_dataloader()`` 分别为 train/validation 创建实例；
            ``__init__`` 内部依次调用 ``_download()`` 和
            ``_read_files_and_tokenize()``，之后实例被传给 PyTorch ``DataLoader``。
        """
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        """把每个配置路径解析成本进程可读取的本地路径。

        输入：
            无显式参数；读取 ``self.parquet_files`` 和 ``self.cache_dir``。

        输出：
            无返回值；原地把 ``self.parquet_files[i]`` 替换为
            ``copy_local_path_from_hdfs()`` 返回的本地路径。

        调用方式：
            仅由 ``RLHFDataset.__init__()`` 调用，并且发生在读取 Parquet 之前。
        """
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        """读取全部 Parquet 并拼接为行级 DataFrame。

        输入：
            无显式参数；遍历 ``self.parquet_files`` 中已经本地化的路径。

        输出：
            无返回值；为实例设置 ``self.dataframe``（``pandas.DataFrame``）。当前版本
            没有在这里执行 tokenization，已注释的长度过滤也不会改变行数。

        调用方式：
            仅由 ``RLHFDataset.__init__()`` 在 ``_download()`` 之后调用；后续
            ``__len__()`` 和 ``__getitem__()`` 都读取 ``self.dataframe``。
        """
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # 官方此版本已注释掉长度过滤，所以打印的 filter len 通常与 original 相同。
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        """返回 DataFrame 行数；由 DataLoader 计算样本数和 batch 数时调用。"""
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        将一行 prompt 转成模型张量，并原样保留 reward/来源等字段。

        输入：
            ``item``（``int``）：DataFrame 的位置索引，由 PyTorch DataLoader/Sampler
            传入。对应行必须包含 ``self.prompt_key`` 指向的 chat prompt；其余列作为
            non-tensor 元数据保留。

        输出：
            ``dict``：保留原行的 ``data_source/ability/reward_model/extra_info`` 等字段，
            并新增：

            - ``input_ids``（``torch.Tensor[int64]``）：shape ``[max_prompt_length]``；
            - ``attention_mask``（``torch.Tensor``）：同 shape，左 padding 为 0；
            - ``position_ids``（``torch.Tensor[int64]``）：同 shape；
            - ``index``：从 ``extra_info['index']`` 提升到顶层，缺失时默认为 0；
            - 可选 ``raw_prompt``：当 ``return_raw_chat=True`` 时保留未套模板的 chat。

        调用方式：
            PyTorch DataLoader 按 sampler 产生的索引逐条调用；一批返回值随后交给
            本文件 ``collate_fn()``。训练器再通过 ``DataProto.from_single_dict()``
            把该 batch 送入 rollout、reward 和 actor update 主链。

        典型 ``chat`` 是 ``[{"role": "user", "content": "..."}]``。有 chat template
        的 tokenizer 会加模型约定的控制 token 和 generation prompt；Base 模型若没有
        template，则直接取第一条消息正文。
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat

        # left_pad=True：不同长度 prompt 的最后一个有效 token 对齐，便于自回归生成。
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        # pad 位置为 0；有效 token 的位置从 0 连续递增。
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # GRPO 要把同一道题的多条 rollout 放在同一组；index 是稳定的题目分组键。
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

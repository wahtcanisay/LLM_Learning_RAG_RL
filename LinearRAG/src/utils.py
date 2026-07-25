from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
import httpx
from openai import OpenAI
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """用“对象类型前缀 + 内容 MD5”生成稳定 ID。

    `passage-`、`entity-`、`sentence-` 等 namespace 可防止不同对象类型因文本
    相同而被误认为同一节点。MD5 在这里用于稳定寻址和去重，不用于安全认证。
    """
    return prefix + md5(content.encode()).hexdigest()

class LLM_Model:
    """最小 OpenAI-compatible LLM 包装器。

    LLM 只用于 `qa()` 生成答案和 `Evaluator` 的 LLM Judge。LinearRAG 的
    relation-free 构图本身不调用它，这正是官方所强调的构图零 LLM token 成本。
    """

    def __init__(self, llm_model):
        # trust_env=False 避免 httpx 自动读取系统代理；服务地址由环境变量显式给出。
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client
        )
        self.llm_config = {
            "model": llm_model,
            "max_tokens": 2000,
            # temperature=0 降低随机性，但不代表输出绝对可复现。
            "temperature": 0,
        }
    def infer(self, messages):
        """发送一组 Chat Completions messages，并返回首个文本答案。"""
        response = self.openai_client.chat.completions.create(**self.llm_config,messages=messages)
        return response.choices[0].message.content



def normalize_answer(s):
    """为宽松字符串包含评测统一大小写、标点、冠词与空白。"""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 
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

def setup_logging(log_file):
    """同时把 INFO 日志输出到终端和本次实验的 UTF-8 文件。"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    """把一组分数线性缩放到 [0, 1]，供不同奖励项组合。

    所有值相等时不存在可用的相对次序，官方实现返回全 1，保留统一先验。
    """
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

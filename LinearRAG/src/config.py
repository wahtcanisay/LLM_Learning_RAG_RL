from dataclasses import dataclass, field
from src.utils import LLM_Model

@dataclass
class LinearRAGConfig:
    """集中保存一次 LinearRAG 构图、检索和生成实验的全部配置。

    阅读时可以把字段分成六组：

    1. 资源身份：dataset/model/spaCy；
    2. 持久化与吞吐：working_dir/batch_size/max_workers；
    3. 最终检索输出：retrieval_top_k；
    4. Entity → Sentence → Entity 传播；
    5. Passage 先验与 Personalized PageRank；
    6. 可选的属性关键词回退。

    特别注意：`use_vectorized_retrieval` 的“向量化”指用 PyTorch 稀疏矩阵
    加速图传播，不等同于 MedRAG 中用 Embedding 排序 Passage 的 Dense Retrieval。
    """

    # 资源身份：同一 dataset_name 也决定缓存文件保存在哪个子目录。
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    # 当前入口直接读取已切好的 chunks；下面两个切块参数在这条 run.py 链路中未使用。
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"

    # 持久化和并行吞吐。
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16

    # 最终交给生成模型的 Passage 数量。
    retrieval_top_k: int = 5

    # 图上的语义桥接传播：最多传播轮数、每个实体挑几句、低于何分停止扩展。
    max_iterations: int = 3
    top_k_sentence: int = 1

    # Passage 的重启先验：Dense 分数占比、写入 PPR 重启向量后的整体缩放，
    # 以及 PPR 沿图继续游走的阻尼系数。
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5

    # False：Python 循环的 BFS-style 分层传播；True：PyTorch 稀疏矩阵传播。
    use_vectorized_retrieval: bool = False  # True for vectorized matrix computation, False for BFS iteration

    # 可选属性问题增强：只有显式开启且问题含 where/when/born 等词时才生效。
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])

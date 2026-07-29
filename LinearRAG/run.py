"""LinearRAG 官方端到端入口，也是阅读整个仓库的第一张“地图”。

真实运行流向：

    ① parse_arguments()
    ② 加载 Embedding 模型、questions/chunks、日志与 LLM 客户端
    ③ 用这些资源组装 LinearRAGConfig，并初始化 LinearRAG
    ④ LinearRAG.index(passages) 执行离线构图
       └─ 可能复用已有 Parquet Embedding、NER JSON 和图数据
    ⑤ LinearRAG.qa(questions) 执行在线检索，再调用 LLM 生成答案
    ⑥ 保存 predictions.json，并由 Evaluator.evaluate() 评测

和已经学过的 MedRAG 相比，这里的主要新增点不是另一个 Dense Retriever，
而是先把 Passage 与 Entity 组织成图，再用实体传播和 Personalized PageRank
为 Passage 排序。本文件只增加学习注释，不改变官方运行行为。
"""

import argparse
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime

# 学习注意：这是官方代码针对其多卡机器写死的 GPU 编号假设。
# 单卡机器正式运行前需要单独设计可配置方案；本轮为保持行为不变，只解释、不修改。
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
warnings.filterwarnings('ignore')

def parse_arguments():
    """读取一次实验所需的模型、数据集和图传播超参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="model/all-mpnet-base-v2", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="novel", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    parser.add_argument("--max_iterations", type=int, default=3, help="The max number of iterations to use")
    parser.add_argument("--iteration_threshold", type=float, default=0.4, help="The threshold for iteration")
    parser.add_argument("--passage_ratio", type=float, default=2, help="The ratio for passage")
    parser.add_argument("--top_k_sentence", type=int, default=3, help="The top k sentence to use")
    parser.add_argument("--use_vectorized_retrieval", action="store_true", help="Use vectorized matrix-based retrieval instead of BFS iteration")
    return parser.parse_args()


def load_dataset(dataset_name): 
    """读取作者预处理好的问答与语料。

    数据目录契约：
        dataset/<dataset_name>/questions.json
        dataset/<dataset_name>/chunks.json

    `questions` 中的每项稍后至少要提供 `question` 和 `answer`；`chunks`
    是可被检索的 Passage 文本列表。
    """
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = f"dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # 把原始顺序写进 Passage 文本。这个前缀不仅用于展示：index() 中的
    # add_adjacent_passage_edges() 会重新解析它，从而连接相邻 Passage。
    # 注意：前缀会参与内容哈希和 Embedding，因此已成为索引文本的一部分。
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)] # 最后的passages得到字符串列表，内涵格式如：["0:chunk1"，"1:chunk2"]
    return questions, passages

def load_embedding_model(embedding_model):
    """把 SentenceTransformer 放到 CUDA，用于三类文本和问题的统一编码。"""
    embedding_model = SentenceTransformer(embedding_model,device="cuda")
    return embedding_model

def main():
    """按“资源准备 → 离线构图 → 在线问答 → 评测”执行完整实验。"""
    # ① 解析实验参数，并为本次运行创建唯一的结果目录时间戳。
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()

    # ② 加载共享资源与数据。Embedding 同时服务 Passage、Entity、Sentence；
    # LLM 只用于最终答案生成和 LLM Judge，不参与 relation-free 图构建。
    embedding_model = load_embedding_model(args.embedding_model)
    questions,passages = load_dataset(args.dataset_name)
    setup_logging(f"results/{args.dataset_name}/{time_str}/log.txt")
    llm_model = LLM_Model(args.llm_model)

    # ③ 将入口参数集中为配置对象，再初始化三个 EmbeddingStore、NER 和无向图。
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,         
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,             #最多进行多少层 Entity → Sentence → Entity 传播
        iteration_threshold=args.iteration_threshold,   #实体传播分数的最低阈值
        passage_ratio=args.passage_ratio,               #最终 Passage 打分时，直接的 Question–Passage Dense 相似度所占权重
        top_k_sentence=args.top_k_sentence,             #每个当前实体最多选择多少个与问题最相似的句子，作为语义桥，继续寻找下一层实体
        use_vectorized_retrieval=args.use_vectorized_retrieval
    )
    rag_model = LinearRAG(global_config=config)

    # ④ 先离线 index()，再在线 qa()。
    # qa() 内部先 retrieve() 得到 Top-k Passage，随后才调用 LLM 生成答案。
    rag_model.index(passages)
    questions = rag_model.qa(questions)

    # ⑤ 保存逐题预测，然后在同一文件中回写逐题指标并生成汇总指标文件。
    os.makedirs(f"results/{args.dataset_name}/{time_str}", exist_ok=True)
    with open(f"results/{args.dataset_name}/{time_str}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    evaluator = Evaluator(llm_model=llm_model, predictions_path=f"results/{args.dataset_name}/{time_str}/predictions.json")
    evaluator.evaluate(max_workers=args.max_workers)
if __name__ == "__main__":
    # 只有直接执行 `python run.py ...` 时才进入完整流程；导入模块不会自动运行。
    main()

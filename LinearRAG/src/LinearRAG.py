from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch
logger = logging.getLogger(__name__)


class LinearRAG:
    """LinearRAG 的离线构图器、在线检索器与问答编排器。

    相对 MedRAG 的 Dense/Hybrid 检索，这个类增加了一层 GraphRAG 结构：

    离线：
        Passage → NER → Entity/Sentence 映射 → 三类 Embedding
        → Entity–Passage 边 + 相邻 Passage 边 → igraph

    在线：
        Question → Seed Entity → Entity→Sentence→Entity 语义传播
        → Dense Passage 先验 → Personalized PageRank → Top-k Passage

    Relation-free（关系无关）表示它不让 LLM 抽取“实体—关系类型—实体”
    三元组，而是依靠轻量 NER、同句桥接和 Passage 邻接建立可传播结构。

    学习注意：当前代码虽维护 Sentence Embedding 和 Entity–Sentence 映射，
    但最终 igraph 的正式顶点只有 Entity 与 Passage；Sentence 是在线传播桥，
    不是 `add_nodes()` 加入的第三类图顶点。
    """

    def __init__(self, global_config):
        """初始化共享模型、三类缓存、NER 与最终无向图。"""
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        retrieval_method = "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")
        
        # 向量化分支会用 GPU 稀疏矩阵做图传播；普通 BFS-style 分支主要用 NumPy。
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")
        
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        # 图是无向的：Entity–Passage 与 Passage–Passage 连接可双向传播权重。
        self.graph = ig.Graph(directed=False)

    def load_embedding_store(self):
        """为 Passage、Entity、Sentence 分别加载可增量复用的 Parquet 缓存。"""
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, 
                                                      db_filename=os.path.join(self.config.working_dir,self.dataset_name, "passage_embedding.parquet"), 
                                                      batch_size=self.config.batch_size, 
                                                      namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, 
                                                     db_filename=os.path.join(self.config.working_dir,self.dataset_name, "entity_embedding.parquet"), 
                                                     batch_size=self.config.batch_size, 
                                                     namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def load_existing_data(self,passage_hash_ids):
        """加载 NER 缓存，并找出这次真正需要做 NER 的新 Passage。

        Cache reuse（缓存复用）使重复运行不必重新处理全部语料。这里以稳定的
        Passage 内容哈希判断新旧，而不是依赖列表位置。
        """
        self.ner_results_path = os.path.join(self.config.working_dir,self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        """先检索证据，再并行调用 LLM 生成答案。

        检索命中与答案正确仍是两件事：本方法消费 Top-k Passage 形成 Prompt，
        但生成模型仍可能忽略证据、推理失败或产生幻觉。
        """
        retrieval_results = self.retrieve(questions)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            # 把排好序的 Passage 原样串成上下文；当前实现没有单独的 token 截断。
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
        # max_workers 控制并发 API 请求数，不是 GPU batch size。
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))

        for qa_result,question_info in zip(all_qa_results,retrieval_results):
            try:
                # 官方 Prompt 约定用 "Answer:" 分隔最终答案；格式不符合时保留原响应。
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results
        
    def retrieve(self, questions):
        """对每个问题执行图检索，必要时回退为纯 Dense Passage Retrieval。

        在线分派：
            Question Embedding + Question NER
            → 有 Seed Entity：实体传播 + Passage 先验 + PPR
            → 无 Seed Entity：直接按 Question–Passage 向量点积排序

        Dense fallback（稠密回退）保证 NER 没找到实体时系统仍能返回证据。
        """
        # 把持久化 Store 展开为本轮批量查询可直接使用的数组和 ID 列表。
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        # igraph 内部使用整数顶点编号；业务层使用稳定 hash ID，故需要双向映射。
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

        # 向量化图传播需要先把 Entity–Sentence 双向映射转换为稀疏邻接矩阵。
        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
            logger.info(f"Device: {self.device}")

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            # Store 中向量已归一化；问题也归一化后，点积可解释为余弦相似度。
            question_embedding = self.config.embedding_model.encode(question,normalize_embeddings=True,
                                                                    show_progress_bar=False,
                                                                    batch_size=self.config.batch_size)
            seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores = self.get_seed_entities(question)
            if len(seed_entities) != 0:
                # 能把问题接入实体空间时，执行 LinearRAG 的核心图检索。
                sorted_passage_hash_ids,sorted_passage_scores = self.graph_search_with_seed_entities(question,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
            else:
                # NER 无结果时无法选 Seed Entity，退回与 MedRAG Dense 类似的 Passage 排序。
                sorted_passage_indices,sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"]
            }
            retrieval_results.append(result)
        return retrieval_results
    
    def _precompute_sparse_matrices(self):
        """预计算向量化传播使用的两张稀疏邻接矩阵。

        稀疏邻接矩阵只存真实存在的边，避免为大量不存在的 Entity–Sentence
        组合分配内存。形状分别是：

            Entity-to-Sentence: [实体数, 句子数]
            Sentence-to-Entity: [句子数, 实体数]

        COO（Coordinate）格式用“非零元素的行列坐标 + 值”构造张量。
        这一步每次 retrieve() 只做一次，而不是每个问题重建。
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # E2S mention matrix：某实体出现在某句子中，则对应位置为 1。
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        
        # S2E 是反向映射，支持从已选句子继续激活同句实体。
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        
        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        
        # 转为 PyTorch COO 稀疏张量；空边集也创建合法的零非零项张量。
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )
        
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )
            
    def graph_search_with_seed_entities(self, question, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        """组合两阶段检索：先激活实体，再计算 Passage 先验并运行 PPR。"""
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        # Entity 与 Passage 权重位于同一个“图顶点长度”的重启向量中。
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices,ppr_sorted_passage_scores

    def run_ppr(self, node_weights):        
        """运行 Personalized PageRank，并只返回 Passage 顶点的排序。

        PPR 与普通 PageRank 的差别在于 personalization/reset vector（个性化
        重启向量）：随机游走重启时，不是均匀回到任意节点，而是更可能回到
        与当前问题相关的 Entity/Passage。`damping` 控制继续沿边传播的概率。

        这也不同于 MedRAG 的 RRF：RRF 融合多个排名列表，PPR 则沿图边扩散
        当前问题的节点权重。
        """
        # NaN 或负值不能作为重启概率，统一裁成 0。
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        # Entity 节点只帮助传播；最终证据必须是可放入 Prompt 的 Passage。
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        """用 BFS-style 循环执行 Entity → Sentence → Entity 语义传播。

        术语：
            active entity：本轮有资格继续扩展的实体；
            tier/hop：实体距 Seed Entity 的传播轮次；
            semantic bridge：用与问题相似的 Sentence 连接同句实体。

        这不是调用标准 BFS API，而是用 `current_entities`/`new_entities`
        实现逐层扩展。阈值负责剪枝，`top_k_sentence` 限制每个实体选几句，
        `max_iterations` 限制最多传播多少轮。
        """
        actived_entities = {}
        # entity_weights 的长度等于全部 igraph 顶点数，只有 Entity 位置会在此赋值。
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1) # (索引，分数， iter)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                # 低分实体不再向外扩展，减少噪声与计算。
                if entity_score < self.config.iteration_threshold: # 前一步打分过低的entity被过滤
                    continue
                # 取出对应的已激活的entity出现的sentence
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                # 在当前实体关联的句子中，选择最符合问题语义的桥。
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence] # 取得前k条分数最高的sentence
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index] # 相似度分数
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        # 传播分数 = 来路实体分数 × 问题与桥接句子的相似度。
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        """用 PyTorch 稀疏矩阵加速 Entity → Sentence → Entity 传播。

        Vectorized retrieval 在这里是“图传播向量化”，不是 Dense Retrieval。
        它试图保持 BFS-style 分支的三项语义：

        - Sentence 去重：已使用的桥接句不再重复传播；
        - 每个 active Entity 独立挑选 Top-k Sentence；
        - 每轮对 Entity 分数做 iteration_threshold 剪枝。

        矩阵乘法主链：
            Entity score vector
            → E2S 转成 Sentence activation
            → 乘 Question–Sentence similarity
            → S2E 转成下一轮 Entity score
        """
        # Initialize entity weights
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # 所有 Question–Sentence 相似度一次算完，后续各轮复用。
        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
        
        # Convert to torch tensors and move to device
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)
        
        # 布尔掩码对应 BFS 分支的 used_sentence_hash_ids 集合。
        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
        
        # Seed Entity 构成第 0 层稀疏分数向量，也是传播起点。
        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()
        
        # Also maintain a dense accumulator for total scores
        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device), 
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))
        
        # Initialize actived_entities
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        
        current_entity_scores_sparse = entity_scores_sparse
        
        # 每次循环对应一层 hop/tier。
        for iteration in range(1, self.config.max_iterations):
            # Convert sparse tensor to dense for threshold operation
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            
            # Apply threshold to current scores
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold, 
                current_entity_scores_dense, 
                torch.zeros_like(current_entity_scores_dense)
            )
            
            # Get non-zero indices for sparse representation
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(nonzero_indices) == 0:
                break
            
            # Extract non-zero values and create sparse tensor
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()
            
            # 第一步：Entity score 经 E2S 邻接矩阵传播到 Sentence。
            # Convert sparse vector to 2D for matrix multiplication
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()
            
            # E @ E2S -> sentence activation scores (sparse @ sparse = dense)
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )
            # Convert to dense before squeeze to avoid CUDA sparse tensor issues
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()
            
            # Apply sentence deduplication: mask out used sentences
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )
            
            # 第二步：为每个 active Entity 单独选与问题最相似的 Top-k Sentence。
            # This matches BFS behavior: each entity independently selects its top-k sentences
            selected_sentence_indices_list = []
            
            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                # Iterate through each active entity
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]
                    
                    # Get sentences connected to this entity from the sparse matrix
                    # entity_to_sentence_sparse shape: (num_entities, num_sentences)
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]  # Get column indices
                    
                    if len(entity_sentence_indices) == 0:
                        continue
                    
                    # Filter out already used sentences
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]
                    
                    if len(available_sentence_indices) == 0:
                        continue
                    
                    # Get sentence similarities (for ranking)
                    sentence_sims = sentence_similarities[available_sentence_indices]
                    
                    # Select top-k sentences based ONLY on sentence similarity (matches BFS line 240)
                    # NOT weighted by entity_score at selection time
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)
                
                # Merge all selected sentences (with deduplication via unique)
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    
                    # Mark selected sentences as used
                    used_sentence_mask[unique_selected_sentences] = True
                    
                    # 句子传播权重 = 实体侧激活 × Question–Sentence 相似度。
                    weighted_sentence_scores = sentence_activation * sentence_similarities
                    
                    # Zero out non-selected sentences
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    # No sentences selected, create zero vector
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:
                # No active entities or top_k_sentence is 0
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            
            # 第三步：加权 Sentence 经 S2E 邻接矩阵传播到下一层 Entity。
            # Convert to sparse for more efficient computation
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()
                
                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )
                # Convert to dense before squeeze to avoid CUDA sparse tensor issues
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
            
            # 同一 Entity 可被多条路径激活，最终分数累计。
            entity_scores_dense += next_entity_scores_dense
            
            # Update actived_entities dictionary (record last trigger like BFS)
            # This matches BFS behavior: unconditionally update for entities above threshold
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                # Unconditionally update to record the last trigger (matches BFS line 252)
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)
            
            # Prepare sparse tensor for next iteration
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values, 
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break
        
        # Convert back to numpy for final processing
        entity_scores_final = entity_scores_dense.cpu().numpy()
        
        # Map entity scores to graph node weights (only for non-zero scores)
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)
        
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        """计算写入 PPR 重启向量的 Passage 先验。

        核心形式：

            passage_score
              = passage_ratio × 归一化 Dense 相似度
              + log(1 + Entity 出现奖励)
              + 可选属性关键词奖励

            Passage 顶点重启权重
              = passage_score × passage_node_weight

        Dense 部分回答“文本语义是否像问题”，Entity 奖励回答“该 Passage 是否
        包含已沿语义桥激活的实体”。tier 越远，实体奖励除以越大的层级数。
        """
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        apply_attribute_boost = (
            self.config.enable_hybrid_attribute_fallback
            and self._is_attribute_query(question)
        )
        question_lower = question.lower()

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    # Seed Entity 的 tier 可能记为 0 或 1；分母至少为 1，避免除零。
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

            # log 压缩高频实体奖励，避免重复出现次数完全支配 Dense 相关性。
            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(question_lower, passage_text_lower)
                if overlap > 0:
                    passage_score += self.config.attribute_keyword_boost * math.log(1 + overlap)

            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        """对全部归一化 Passage Embedding 做点积并降序排序。

        它既是图检索中的 Passage 语义先验，也是无 Seed Entity 时的回退检索。
        当前实现是 NumPy 全量精确计算，并未使用 FAISS 等 ANN 索引。
        """
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def _is_attribute_query(self, question):
        """判断问题是否含配置中的属性词，如 where、when、born。"""
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(keyword in tokens for keyword in self.config.attribute_query_keywords)

    def _attribute_keyword_overlap(self, question_lower, passage_text_lower):
        """统计同时出现在问题与 Passage 中的属性关键词数量。"""
        overlap = 0
        for keyword in self.config.attribute_query_keywords:
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap
    
    def get_seed_entities(self, question):
        """把 Question Entity 语义对齐到语料 Entity，得到传播种子。

        Seed Entity（种子实体）是问题进入图空间的起点。流程是：

            Question NER
            → 每个问题实体的归一化 Embedding
            → 与全部语料 Entity Embedding 点积
            → 为每个问题实体选分数最高的一个语料 Entity

        这与 Dense Passage Retrieval 不同：这里匹配的是 Entity 空间，输出用于
        图传播而非直接作为最终证据。当前代码没有设置最低相似度阈值，因此即使
        最佳匹配较弱，也会选择一个 Seed Entity；这是需要后续实验验证的边界。
        """
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [],[],[],[]
        question_entity_embeddings = self.config.embedding_model.encode(question_entities,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
        # 形状：[语料 Entity 数, 问题 Entity 数]。
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []       
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx] # 第idx个queryentity的所有分数
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        """离线构建 relation-free 检索结构并保存 GraphML。

        七个阶段：
            ① 编码/持久化 Passage；
            ② 复用 NER 缓存，只处理新 Passage；
            ③ 整理 Entity、Sentence 与双向映射；
            ④ 编码/持久化 Entity 和 Sentence；
            ⑤ 把文本映射转换为稳定 hash-ID 映射；
            ⑥ 建 Entity–Passage 与相邻 Passage 边；
            ⑦ 添加正式图节点/边并保存 GraphML。
        """
        # 临时边表：外层 key 是起点 hash ID，内层保存邻点与边权。
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)

        # ① Passage Store 只编码缺失文本。
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text() #　hashid->passage映射

        # ② 已有 NER JSON 可复用，只对新 Passage 调用 spaCy。
        existing_passage_hash_id_to_entities,existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities,new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)

        # ③ 从 NER 结果得到去重节点集合和 Entity↔Sentence 文本映射。
        entity_nodes, sentence_nodes,passage_hash_id_to_entities,self.entity_to_sentence,self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)

        # ④ Sentence 和 Entity 也各自拥有可复用的 Embedding Store。
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))

        # ⑤ 在线传播使用稳定 ID，而不是重复保存长文本。
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        # ⑥ 当前最终图的两类边：Entity–Passage，以及按原始顺序相邻的 Passage。
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()

        # ⑦ 把临时统计真正写入 igraph，并持久化为通用 GraphML 文件。
        self.augment_graph()
        output_graphml_path = os.path.join(self.config.working_dir,self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)   
        self.graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self):
        """按 `load_dataset()` 写入的数字前缀连接相邻 Passage。

        这类边保存原文局部上下文顺序，权重固定为 1.0。没有数字前缀的文本不会
        进入这条邻接边构建逻辑。
        """
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        """先添加 Entity/Passage 顶点，再批量添加正式边。"""
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        """把 Entity 和 Passage 加为 igraph 顶点。

        关键辨析：Sentence Store 没有合并进 `all_hash_id_to_text`，因此 Sentence
        虽然在在线 Entity 传播中充当 semantic bridge，却不是最终 igraph 顶点。
        """
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        """把临时 `node_to_node_stats` 转成 igraph 边及 `weight` 属性。"""
        edges = []
        weights = []
        
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        """建立 Entity–Passage 边，并按 Passage 内实体总出现次数归一化。

        对某 Passage 中的某 Entity：

            edge_weight
              = 该 Entity 在 Passage 中的字符串出现次数
                / 所有抽取 Entity 在该 Passage 中的总出现次数

        这是无需 LLM 关系抽取的共现连接，也是 relation-free 的关键组成。
        """
        passage_to_entity_count ={} 
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity) # 字符串匹配entity出现次数
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count # set充当key来代表边
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        """把 NER JSON 整理为集合与 Entity↔Sentence 文本映射。

        函数名中的 `edges` 容易误导：此处生成的 Entity–Sentence 关系用于在线
        语义桥接，并不会直接成为 `self.graph` 中的正式边；正式边由后续
        `add_entity_to_passage_edges()` 和 `add_adjacent_passage_edges()` 产生。
        """
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence,entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        """把新 Passage 的 NER 结果并入已有缓存字典。"""
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        """保存 NER 中间结果，供下次增量构图复用。"""
        with open(self.ner_results_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f)

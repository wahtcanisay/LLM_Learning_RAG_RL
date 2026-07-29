from copy import deepcopy
from src.utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os

class EmbeddingStore:
    """面向一种文本类型的增量 Embedding 缓存。

    LinearRAG 分别为 Passage、Entity、Sentence 创建实例。`namespace` 会加入
    内容哈希前缀，所以相同字符串作为不同对象类型时不会共用 ID。

    五个并行数据结构必须保持对齐：
        hash_ids：稳定 ID 的有序列表；
        texts：与 hash_ids 同位置的原文；
        embeddings：与 hash_ids 同位置的向量；
        hash_id_to_idx：ID → 向量行号；
        hash_id_to_text / text_to_hash_id：ID 与原文的双向查询。

    这不是 FAISS：数据量被直接保存在 Parquet 中，检索时由上层 NumPy 点积。
    """

    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace

        # 三个列表存储的一个text的三种表示
        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        # 互相转换的dict
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}
        
        self._load_data()
    
    def _load_data(self):
        """若 Parquet 已存在，就恢复列表和三套查找映射，避免重复编码。"""
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")
        
    def insert_text(self, text_list):
        """只编码缓存中尚不存在的文本，再把新记录追加到持久化文件。"""
        # Content hash（内容哈希）把文本稳定映射为 ID；命名空间负责区分对象类型。
        nodes_dict = {}
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}
        
        all_hash_ids = list(nodes_dict.keys())
        
        # Incremental indexing（增量建索引）：已有 ID 不再调用 Embedding 模型。
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]      
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        # 向量被 L2 归一化后，两个向量的点积等价于余弦相似度。
        all_embeddings = self.embedding_model.encode(texts_to_encode,
                                                     normalize_embeddings=True, 
                                                     show_progress_bar=False,
                                                     batch_size=self.batch_size)
        
        self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def _upsert(self, hash_ids, texts, embeddings):
        """追加三组对齐数据，重建映射，并立即落盘。"""
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
        
        self._save_data()

    def _save_data(self):
        """以一行一个文本对象的形式保存 ID、原文和向量。"""
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings
        })
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)
      
    def get_hash_id_to_text(self):
        """返回深拷贝，避免调用方无意修改 Store 内部映射。"""
        return deepcopy(self.hash_id_to_text)
    
    def encode_texts(self, texts):
        """编码临时查询文本，但不写入缓存。"""
        return self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)
    
    def get_embeddings(self, hash_ids):
        """按稳定 ID 恢复对应向量，返回顺序与输入 hash_ids 一致。"""
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]
        return embeddings

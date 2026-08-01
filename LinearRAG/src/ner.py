import spacy
from collections import defaultdict
import pdb


class SpacyNER:
    """用 spaCy/SciSpaCy 把 Passage 和 Question 连接到 Entity。

    NER（Named Entity Recognition，命名实体识别）用于找出人名、地点、疾病等
    实体提及。离线阶段输出两个关键契约：

        passage_hash_id_to_entities：
            Passage 哈希 ID → 该 Passage 中去重后的 Entity 文本
        sentence_to_entities：
            Sentence 原文 → 同一句中出现的 Entity 文本

    第二套映射让 Sentence 成为 semantic bridge（语义桥）：在线检索可以从当前
    Entity 找到与问题相关的 Sentence，再激活同句的其他 Entity。
    """

    def __init__(self,spacy_model):
        # 通过spacy.load()加载 NLP Pipeline，负责分词、分句和实体识别
        self.spacy_model = spacy.load(spacy_model)

    def batch_ner(self, hash_id_to_passage, max_workers):
        """批量处理 Passage，并合并为全语料级的两套 NER 映射。"""
        passage_list = list(hash_id_to_passage.values()) # 得到chunk文本的list
        # 学习注意：当 Passage 数少于 max_workers 时，这个整数除法可能得到 0。
        # spaCy 是否接受该值需要用 toy 数据验证；本轮只注释，不改变官方行为。
        batch_size = len(passage_list) // max_workers
        docs_list = self.spacy_model.pipe(passage_list,batch_size=batch_size) # 返回的doc应该含有分好的entity和sentence
        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)
        for idx,doc in enumerate(docs_list): # doc.ents 是一个由 spaCy Span 对象组成的序列。每个 ent 代表一次被识别出的实体提及
            passage_hash_id = list(hash_id_to_passage.keys())[idx] # 按顺序拿到passageid
            single_passage_hash_id_to_entities,single_sentence_to_entities = self.extract_entities_sentences(doc,passage_hash_id)
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            for sent, ents in single_sentence_to_entities.items():
                for e in ents:
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
        return passage_hash_id_to_entities,sentence_to_entities # 返回passagehashid->entity / sentence->entity 映射
            
    def extract_entities_sentences(self, doc,passage_hash_id):
        """从一个已解析 Doc 中提取 Passage→Entity 与 Sentence→Entity。"""
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = {}
        # pdb.set_trace()  # 注释掉调试断点
        for ent in doc.ents:
            # 序数和基数通常数量多、区分度低，官方实现不把它们作为图实体。
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            unique_entities.add(ent_text)
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities,sentence_to_entities

    def question_ner(self, question: str):
        """抽取问题实体，作为在线检索寻找 Seed Entity 的起点。

        返回值统一为小写以减少表面形式差异；后续并非只做字符串精确匹配，
        而是编码这些问题实体，并在语料 Entity Embedding 中寻找最近邻。
        """
        doc = self.spacy_model(question)
        question_entities = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(ent.text.lower())
        return question_entities

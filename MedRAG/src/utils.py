
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import faiss
import json
import torch
import tqdm
import numpy as np

# 语料库别名 → 实际 chunk 目录名。MedCorp/MedText 会组合多个语料源。
corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}

# 检索器别名 → 后端名称；RRF-2/RRF-4 表示多个检索器结果的融合组合。
retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["facebook/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
    "RRF-2": ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4": ["bm25", "facebook/contriever", "allenai/specter", "ncbi/MedCPT-Query-Encoder"]
}

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"

    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        """
        Creates a simple Transformer + CLS Pooling model and returns the modules
        """
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        token = kwargs.get('token', None)
        cache_folder = kwargs.get('cache_folder', None)
        revision = kwargs.get('revision', None)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        if 'token' in kwargs or 'cache_folder' in kwargs or 'revision' in kwargs or 'trust_remote_code' in kwargs:
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
                tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


def embed(chunk_dir, index_dir, model_name, **kwarg):
    # Dense Retrieval 的离线阶段：逐个 JSONL 分片编码成向量并保存为 .npy。
    save_dir = os.path.join(index_dir, "embedding")
    
    # Contriever 使用标准 SentenceTransformer；其他模型走自定义 CLS pooling。
    if "contriever" in model_name:
        model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    # fnames为引出chunk之后的jsonl文件(其根目录在chunk_dir当中)
    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            # save_path应为xxx.npy
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue
            texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
            if "specter" in model_name.lower():
                texts = [model.tokenizer.sep_token.join([item["title"], item["content"]]) for item in texts]
            elif "contriever" in model_name.lower():
                texts = [". ".join([item["title"], item["content"]]).replace('..', '.').replace("?.", "?") for item in texts]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts]
            else:
                texts = [concat(item["title"], item["content"]) for item in texts]
            # 不同模型需要不同输入格式，但输出统一为二维 embedding 矩阵。
            embed_chunks = model.encode(texts, **kwarg)
            np.save(save_path, embed_chunks) # 保存路径为index_dir/embedding/xxx.npy
        embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]

def construct_index(index_dir, model_name, h_dim=768, HNSW=False, M=32):
    # Dense Retrieval 的索引阶段：.npy格式的向量写入 FAISS，同时记录向量行号对应的原始 JSONL 分片。
    # 这里学习的是 FAISS 的调用契约，不需要阅读 FAISS 内部 C++ 实现：
    # add() 写入向量，search() 查询，write_index()/read_index() 负责持久化。
    with open(os.path.join(index_dir, "metadatas.jsonl"), 'w') as f:
        f.write("")
    
    if HNSW:
        M = M
        # HNSW 是近似近邻索引；不启用时使用 Flat 精确扫描，适合小规模 toy 数据验证，一种图搜索方式
        # HMSW图快速找到候选向量，再用原始向量计算距离或者相似度，只访问部分候选节点
        # SPECTER 使用 L2；其他当前配置的 Dense 模型使用 Inner Product。
        if "specter" in model_name.lower():
            index = faiss.IndexHNSWFlat(h_dim, M)
        else:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        # h_dim 必须与 embedding 的最后一维一致，否则 FAISS 无法接收向量。
        if "specter" in model_name.lower():
            # L2精确引索，L2越小越相近
            index = faiss.IndexFlatL2(h_dim)
        else:
            # 内积精确引索，比较内积相似度
            index = faiss.IndexFlatIP(h_dim)

    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        # FAISS 的内部位置是全局追加顺序；metadata 必须按完全相同的顺序写入。
        # 例如向量位置 17 对应 metadata[17]，再由 source/index 找回 JSONL 行。
        index.add(curr_embed)
        with open(os.path.join(index_dir, "metadatas.jsonl"), 'a+') as f:
            f.write("\n".join([json.dumps({'index': i, 'source': fname.replace(".npy", "")}) for i in range(len(curr_embed))]) + '\n')

    # faiss.index 保存向量索引，metadatas.jsonl 保存“向量位置 → 文档位置”的映射。
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index


class Retriever: 

    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="textbooks", db_dir="./corpus", HNSW=False, **kwarg):
        # Retriever 初始化可能产生外部副作用：下载语料、下载预计算 embedding 或构建索引。
        # 所以第一次学习只读代码时不要随意实例化 rag=True 的 MedRAG。
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name

        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        if not os.path.exists(self.chunk_dir):
            # 本地没有 chunk 目录时，官方实现会从 Hugging Face 克隆语料。
            print("Cloning the {:s} corpus from Huggingface...".format(self.corpus_name))
            os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus_name, os.path.join(self.db_dir, self.corpus_name)))
            if self.corpus_name == "statpearls":
                print("Downloading the statpearls corpus from NCBI bookshelf...")
                os.system("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(os.path.join(self.db_dir, self.corpus_name)))
                os.system("tar -xzvf {:s} -C {:s}".format(os.path.join(db_dir, self.corpus_name, "statpearls_NBK430685.tar.gz"), os.path.join(self.db_dir, self.corpus_name)))
                print("Chunking the statpearls corpus...")
                os.system("python src/data/statpearls.py")
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name.replace("Query-Encoder", "Article-Encoder"))
        # index_dir 按“语料库 + 文档编码器”隔离，避免不同模型的向量混用。
        # BM25 分支使用 Pyserini/Lucene 倒排索引，不需要生成 embedding。
        if "bm25" in self.retriever_name.lower():
            # LuceneSearcher是Pyserini提供的BM25搜索器封装，负责打开建立好的Lucene索引，接受字符串查询，计算相关性，返回top-k文档及其分数
            from pyserini.search.lucene import LuceneSearcher
            self.metadatas = None
            self.embedding_function = None
            if os.path.exists(self.index_dir):
                self.index = LuceneSearcher(os.path.join(self.index_dir))
            else:
                # 不存在索引库时调用 Pyserini 的命令行封装构建 Lucene 索引；
                # 我们只需理解输入/输出和检索接口，不需通读 Lucene 源码。
                os.system("python -m pyserini.index.lucene "
                "--collection JsonCollection "
                "--input {:s} --index {:s} " # 读取input中的jsonl文件，并将每个chunk转换为Lucene文档，建立词项->文档倒排映射，文档频率/长度，以及BM25的引索结构
                "--generator DefaultLuceneDocumentGenerator "
                "--threads 16".format(self.chunk_dir, self.index_dir))
                self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            # Dense 分支读取或构建 FAISS 索引，并加载查询编码器。
            if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
                # 已有索引直接加载；metadata 与 FAISS 的向量位置一一对应。
                self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            else:
                print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                if self.corpus_name in ["textbooks", "pubmed", "wikipedia"] and self.retriever_name in ["allenai/specter", "facebook/contriever", "ncbi/MedCPT-Query-Encoder"] and not os.path.exists(os.path.join(self.index_dir, "embedding")):
                    #已有计算好的embedding则下载
                    print("[In progress] Downloading the {:s} embeddings given by the {:s} model...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                    os.makedirs(self.index_dir, exist_ok=True)
                    if self.corpus_name == "textbooks":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EYRRpJbNDyBOmfzCOqfQzrsBwUX0_UT8-j_geDPcVXFnig?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQqzldVMCCVIpiFV4goC7qEBSkl8kj5lQHtNq8DvHJdAfw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    elif self.corpus_name == "pubmed":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ebz8ySXt815FotxC1KkDbuABNycudBCoirTWkKfl8SEswA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EWecRNfTxbRMnM0ByGMdiAsBJbGJOX_bpnUoyXY9Bj4_jQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EVCuryzOqy5Am5xzRu6KJz4B6dho7Tv7OuTeHSh3zyrOAw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    elif self.corpus_name == "wikipedia":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ed7zG3_ce-JOmGTbgof3IK0BdD40XcuZ7AGZRcV_5D2jkA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/ETKHGV9_KNBPmDM60MWjEdsBXR4P4c7zZk1HLLc0KVaTJw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EXoxEANb_xBFm6fa2VLRmAcBIfCuTL-5VH6vl4GxJ06oCQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    os.system("unzip {:s} -d {:s}".format(os.path.join(self.index_dir, "embedding.zip"), self.index_dir))
                    os.system("rm {:s}".format(os.path.join(self.index_dir, "embedding.zip")))
                    h_dim = 768
                else:
                    # 没有则使用embed()
                    h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), **kwarg)
                # 构建索引，生成faiss.index /metadata.jsonl
                print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
                self.index = construct_index(index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), h_dim=h_dim, HNSW=HNSW)
                print("[Finished] Corpus indexing finished!")
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]            
            if "contriever" in self.retriever_name.lower():
                self.embedding_function = SentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, id_only=False, **kwarg):
        assert type(question) == str
        question = [question]

        # 查询阶段再次分成 BM25 与 Dense 两条路径：前者直接用词项搜索，后者先编码 query。
        if "bm25" in self.retriever_name.lower():
            # bm25分支能从docid中直接解析出原始的Jsonl文件名+该Jsonl文件当中的行号
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            res_[0].append(np.array([h.score for h in hits])) # h.score放入res_方便后续一共处理
            ids = [h.docid for h in hits]
            indices = [{"source": '_'.join(h.docid.split('_')[:-1]), "index": eval(h.docid.split('_')[-1])} for h in hits]
        else:
            with torch.no_grad():
                # query 必须使用与文档向量兼容的编码器和 pooling 方式。
                query_embed = self.embedding_function.encode(question, **kwarg)
            # FAISS search 返回 (scores/distances, indices)；这里的 indices 不是 JSONL 行号，
            # 而是 FAISS 向量位置，必须先经过 self.metadatas 才能得到 source/index。
            res_ = self.index.search(query_embed, k=k)
            ids = ['_'.join([self.metadatas[i]["source"], str(self.metadatas[i]["index"])]) for i in res_[1][0]]
            indices = [self.metadatas[i] for i in res_[1][0]] # res_[1][0]为当前 query 的 FAISS 全局向量 ID/ res_[0][0]为当前 query 的分数或距离

        scores = res_[0][0].tolist()
        
        if id_only:
            return [{"id":i} for i in ids], scores
        else:
            return self.idx2txt(indices), scores

    def idx2txt(self, indices): # return List of Dict of str
        '''
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of str
        '''
        return [json.loads(open(os.path.join(self.chunk_dir, i["source"]+".jsonl")).read().strip().split('\n')[i["index"]]) for i in indices]

class RetrievalSystem:

    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", HNSW=False, cache=False):
        # RetrievalSystem 是 MedRAG 与具体 Retriever 之间的统一门面。
        # 一个别名可能展开成多个 Retriever × 多个 corpus，下面的双层循环负责实例化它们。
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None
    
    def retrieve(self, question, k=32, rrf_k=100, id_only=False):
        '''
            Given questions, return the relevant snippets from the corpus
        '''
        assert type(question) == str

        output_id_only = id_only
        if self.cache:
            id_only = True

        texts = []
        scores = []

        # 融合检索时先扩大候选池，再由 merge() 统一排序后截回 k 条。
        if "RRF" in self.retriever_name:
            k_ = max(k * 2, 100)
        else:
            k_ = k
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_, id_only=id_only)
                texts[-1].append(t)
                scores[-1].append(s)
        # texts/scores 的形状是“检索器 × 语料库”；merge 会把它们合并为最终结果。
        texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
        if self.cache:
            texts = self.docExt.extract(texts)
        return texts, scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        '''
            Merge the texts and scores from different retrievers
        '''
        # Reciprocal Rank Fusion：同一文档在不同结果列表中排名越靠前，累计分越高。
        RRF_dict = {}
        for i in range(len(retriever_names[self.retriever_name])):
            texts_all, scores_all = None, None
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]
            # SPECTER 的 FAISS 分数是距离，越小越相关；其他分数按越大越相关排序。
            if "specter" in retriever_names[self.retriever_name][i].lower():
                sorted_index = np.array(scores_all).argsort()
            else:
                sorted_index = np.array(scores_all).argsort()[::-1]
            texts[i] = [texts_all[i] for i in sorted_index]
            scores[i] = [scores_all[i] for i in sorted_index]
            for j, item in enumerate(texts[i]):
                # 以文档 id 去重：重复出现的文档只保留一个条目，但继续累加 RRF 分数。
                if item["id"] in RRF_dict:
                    RRF_dict[item["id"]]["score"] += 1 / (rrf_k + j + 1)
                    RRF_dict[item["id"]]["count"] += 1
                else:
                    RRF_dict[item["id"]] = {
                        "id": item["id"],
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "score": 1 / (rrf_k + j + 1),
                        "count": 1
                        }
        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        # 单检索器直接截断；多检索器使用 RRF 分数排序后输出统一字段。
        if len(texts) == 1:
            texts = texts[0][:k]
            scores = scores[0][:k]
        else:
            texts = [dict((key, item[1][key]) for key in ("id", "title", "content")) for item in RRF_list[:k]]
            scores = [item[1]["score"] for item in RRF_list[:k]]
        return texts, scores
    

class DocExtracter:
    
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
        self.db_dir = db_dir
        self.cache = cache
        print("Initializing the document extracter...")
        for corpus in corpus_names[corpus_name]:
            if not os.path.exists(os.path.join(self.db_dir, corpus, "chunk")):
                print("Cloning the {:s} corpus from Huggingface...".format(corpus))
                os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus, os.path.join(self.db_dir, corpus)))
                if corpus == "statpearls":
                    print("Downloading the statpearls corpus from NCBI bookshelf...")
                    os.system("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(os.path.join(self.db_dir, corpus)))
                    os.system("tar -xzvf {:s} -C {:s}".format(os.path.join(self.db_dir, corpus, "statpearls_NBK430685.tar.gz"), os.path.join(self.db_dir, corpus)))
                    print("Chunking the statpearls corpus...")
                    os.system("python src/data/statpearls.py")
        if self.cache:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            _ = item.pop("contents", None)
                            # assert item["id"] not in self.dict
                            self.dict[item["id"]] = item
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"])), 'w') as f:
                    json.dump(self.dict, f)
        else:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            # assert item["id"] not in self.dict
                            self.dict[item["id"]] = {"fpath": os.path.join(corpus, "chunk", fname), "index": i}
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"])), 'w') as f:
                    json.dump(self.dict, f, indent=4)
        print("Initialization finished!")
    
    def extract(self, ids):
        if self.cache:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(item)
        else:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(json.loads(open(os.path.join(self.db_dir, item["fpath"])).read().strip().split('\n')[item["index"]]))
        return output

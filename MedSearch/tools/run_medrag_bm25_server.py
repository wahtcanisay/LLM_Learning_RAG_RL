import os
import uvicorn

from Search-R1.search_r1.search.retrieval_server import Config, app, get_retriever

config = Config(
    retrieval_method="bm25",
    index_path=os.environ.get("MEDSEARCH_MEDRAG_INDEX", "MedSearch/data/retrieval/medrag_textbooks/bm25"),
    corpus_path=os.environ.get("MEDSEARCH_MEDRAG_CORPUS", "MedSearch/data/retrieval/medrag_textbooks.jsonl"),
    retrieval_topk=int(os.environ.get("MEDSEARCH_TOPK", "5")),
    faiss_gpu=False,
    retrieval_model_path="",
    retrieval_pooling_method="mean",
    retrieval_query_max_length=256,
    retrieval_use_fp16=False,
    retrieval_batch_size=128,
)
get_retriever(config)
uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MEDSEARCH_PORT", "8001")))

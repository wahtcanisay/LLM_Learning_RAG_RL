# Luna：MedSearch BM25 50 条审计

审计日期：2026-09-09。未训练模型、未下载完整 PubMed、未修改核心代码和
`STUDY_PROGRESS.md`。

## 结论

本次优先审计了 MedSearch 的真实 BM25 rollout 产物，而不是只看基础阶段的
MedicalGraphRAG 指标。`MedSearch/outputs/dapo_epoch1_bm25.log` 记录了 BM25 服务在
`0.0.0.0:8000` 启动，并有大量 `/retrieve` HTTP 200；对应的
`MedSearch/outputs/dapo_epoch1_n5_s42/rollouts/*.jsonl` 保存了返回的
`<information>`。

按 rollout 文件编号 1--50 抽样，gold 来自同一项目的
`MedSearch/data/rl/train.parquet` / `dev.parquet` 的 `golden_answers`，检查 top-5 返回
信息是否直接包含 gold 答案文本：

- 50/50 rollout 成功匹配到本地 RL 数据中的问题和 gold。
- 4/50 的 top-5 information 直接包含 gold 文本（严格 lexical support，8%）。
- 46/50 不包含 gold 文本；其中 12 条 rollout 的结果解析出 0 个 Doc，说明实际轨迹中
  没有形成可供模型阅读的 information（需结合完整 output 判断是未检索、截断还是协议失败）。
- 这不是“答案正确率”：只证明返回证据中是否出现标准答案字符串。医学同义表达、推理可支持
  但不出现答案原词的情况会被严格规则记为 false。

这组结果支持当前假设：**MedSearch 中出现答案错误，至少有一部分并非生成器单独出错，而是
top-5 没有把 gold 或足够直接的证据带进 context；但仅凭这 50 条，不能把所有错误归因于
retriever。**

## 1. 50 条真实 rollout 审计结果

`strict=Y` 表示 gold 字符串在 `<information>` 的 top-5 文本中出现；`docs` 是从保存的
information 中解析出的 Doc 数。标题是实际返回标题的简写记录。

| # | rollout | sample id | gold | docs | strict | 返回标题（按顺序） |
|---:|---:|---|---|---:|:---:|---|
| 1 | 1 | 30630 | Barrier method | 5 | N | Female sex tourism; Female condom; Bareback; Migrant sex work; excerpt demographic factors |
| 2 | 2 | 4923 | Trastuzumab | 4 | Y | Male breast cancer; Metastatic breast cancer; MammaPrint; Celltrion |
| 3 | 3 | 12683 | Disulphide bond | 4 | N | Alpha-keratin; Alpha-keratin; Fibril; Natural fiber |
| 4 | 4 | 34530 | Antiplatelet aggregatory | 4 | N | Stroke; Mechanism of action of aspirin; Adenosine diphosphate receptor inhibitor; Aspirin |
| 5 | 5 | 12406 | Candida albicans | 0 | N | — |
| 6 | 6 | 10392 | Oral sildenafil citrate trial | 0 | N | — |
| 7 | 7 | 20987 | Vogt–Koyanagi–Harada disease, Sarcoidosis | 3 | N | Granulomatosis with polyangiitis; Granuloma; Granuloma |
| 8 | 8 | 30148 | Triangular wedge | 0 | N | — |
| 9 | 9 | 34841 | Tyrosine kinase | 4 | Y | Fibroblast growth factor receptor 1; Fibroblast growth factor; receptor 3; receptor 4 |
| 10 | 10 | 6060 | Liver | 2 | N | TP53-inducible glycolysis and apoptosis regulator; Fructose 1,6-bisphosphatase |
| 11 | 11 | 34811 | Carbamoyl phosphate synthetase I | 6 | Y | Carbamoyl phosphate; Ornithine transcarbamylase deficiency; repeated Carbamoyl phosphate |
| 12 | 12 | 21964 | 66 liters per minute | 4 | N | Vital capacity; Pulmonary function testing; Respiratory arrest; Vital capacity |
| 13 | 13 | 34491 | No hepatitis virus causes persistent marrow aplasia. | 0 | N | — |
| 14 | 14 | 32295 | Sulphuric acid | 5 | N | Gastric mucosa; Gastric mucosa; Bowman's capsule; Olfactory mucosa; Pierina |
| 15 | 15 | 15265 | Prostaglandin analogue | 3 | N | Diabetic retinopathy; Diabetes insipidus; Bevacizumab |
| 16 | 16 | 10269 | Intravenous normal saline | 5 | N | Adult T-cell leukemia/lymphoma; PTH-related protein; Hypercalcaemia; PTH-related reference; references |
| 17 | 17 | 17709 | On return from dialysis | 5 | N | Enalaprilat; ACE inhibitor; Lincomycin; Abciximab; Renal Monitoring |
| 18 | 18 | 23537 | Normal B cell count | 0 | N | — |
| 19 | 19 | 17361 | Locard's principle | 9 | N | Russian Roulette; John Spenkelink; Ballot Box Bunny; Bernie; unrelated pages |
| 20 | 20 | 8573 | Diagnosis is readily made with characteristic metabolic response to rifampin. | 0 | N | — |
| 21 | 21 | 15090 | Accelerated idioventricular rhythm | 0 | N | — |
| 22 | 22 | 30432 | Psoas major | 0 | N | — |
| 23 | 23 | 26465 | Dark ground microscopy of ulcer discharge | 5 | N | Chancre; Condom effectiveness; Genital ulcer; repeated Genital ulcer |
| 24 | 24 | 11358 | Metalloproteinase-mediated protein degradation | 0 | N | — |
| 25 | 25 | 18259 | Oral sildenafil titrate trial | 0 | N | — |
| 26 | 26 | 32795 | Pulmonary ejection sound | 0 | N | — |
| 27 | 27 | 36280 | Adrenergic a2 receptors inhibit, while b1 receptors stimulate adenyl cyclase. | 0 | N | — |
| 28 | 28 | 6374 | Extensor pollicis longus | 5 | N | Extensor carpi radialis brevis; repeated Cozen's test |
| 29 | 29 | 18029 | Mullerian agenesis | 5 | N | XX male syndrome; XY sex-determination; Gonadal dysgenesis; repeated XX male |
| 30 | 30 | 33299 | Hyaloideocapsular ligament of Weiger | 0 | N | — |
| 31 | 31 | 11375 | b rays | 0 | N | — |
| 32 | 32 | 34372 | Southamptom grading system | 0 | N | — |
| 33 | 33 | 9226 | CIN-1 | 4 | N | Anal Pap smear; Pap test; Bethesda system; cervical intraepithelial neoplasia |
| 34 | 34 | 11664 | Water intoxication | 0 | N | — |
| 35 | 35 | 22228 | Elevated pulmonary artery pressure | 5 | N | Lung cancer; VATS lobectomy; repeated VATS lung resection |
| 36 | 36 | 33969 | Fissure of Santorini | 4 | N | Auriculotemporal nerve; Otic ganglion; Frey's syndrome; Neural iconic strategies |
| 37 | 37 | 2043 | Platelet adhesion | 0 | N | — |
| 38 | 38 | 18964 | Tobacco | 0 | N | — |
| 39 | 39 | 33109 | Absence of oligoclonal band in CSF | 0 | N | — |
| 40 | 40 | 30133 | Streptococcus pneumoniae | 5 | N | Congenital CMV infection; Human CMV; CMV retinitis; repeated Human CMV |
| 41 | 41 | 22808 | Pulmonary trunk and ascending aorta | 4 | N | Pericardial window; Cardiac tamponade; Pericardial fluid; Coronary sinus |
| 42 | 42 | 29912 | Vibrio vulnificus | 0 | N | — |
| 43 | 43 | 11421 | Wayson staining | 5 | N | Inflammation; Mycobacterial cervical lymphadenitis; Connectome; Acrosome reaction; unrelated page |
| 44 | 44 | 28171 | Dexamethasone | 3 | Y | Roxibolone; Glucocorticoid; selective glucocorticoid receptor modulator |
| 45 | 45 | 36365 | Prednisone | 3 | N | Glucocorticoid; Glatiramer acetate; repeated Glatiramer acetate |
| 46 | 46 | 35731 | 4 to 6 months | 0 | N | — |
| 47 | 47 | 36612 | Type II | 0 | N | — |
| 48 | 48 | 34177 | 18 + 6 months | 4 | N | Tooth decay; Day One; Cariogram; Dentistry for babies |
| 49 | 49 | 31802 | Thyroid scan | 0 | N | — |
| 50 | 50 | 1053 | Ecological | 5 | N | Case–control study; Glyphosate; Efficacy of prayer; oil spill; occupationally exposed |

## 2. 已确认的 MedSearch 检索路径与阻塞

- Agent 配置：`MedSearch/configs/medsearch_agent.yaml`，`top_k: 5`，URL 为
  `http://127.0.0.1:8000/retrieve`。
- HTTP 客户端：`MedSearch/rl/retriever_client.py`，请求体为
  `{"queries": [query], "topk": 5, "return_scores": true}`。
- 本次检查时 host 的 8000 端口为 **connection refused**；因此不能现场重放请求。
- 容器 `llm-pytorch` 可访问当前工作区，但容器没有 `curl`；其 Python/Pyserini 可用。
- 已确认服务历史日志：`MedSearch/outputs/dapo_epoch1_bm25.log`，含启动和 `/retrieve` 200 日志。
- 已确认真实语料：`MedSearch/data/stage_b/medtext_corpus.jsonl`，497,498 行，约 746 MB。
  字段包括 `id`、`document_id`、`source`、`source_file`、`source_id`、`source_line`、
  `title`、`content`、`contents`；示例来源是 `statpearls`。
- 未发现 MedSearch 当前 checkout 中对应的持久 BM25 Lucene index。历史 rollout 可审计，
  但无法仅凭当前文件重启同一个服务并验证 index hash。
- 直接在容器对 746 MB 语料做临时流式 BM25 扫描，第一遍真实统计完成：497,498 docs，
  平均 91.67 tokens，约 21.8 秒；第二遍在容器资源限制下被中止，没有把未完成扫描当作结果。

因此本报告的 50 条 MedSearch 结果来自**真实已保存的 BM25 服务返回**，不是临时近似 BM25
计算，也不是编造；当前阻塞点是服务已退出且持久 index 未随工作区保留。

## 3. 与基础 MedicalGraphRAG BM25 的关系

另有可复用的基础检索产物：

- 数据：`MedicalGraphRAG/data/processed/pubmedqa_hard_v1/`
  - 1,000 questions，5,000 documents，7,562 chunks，1,000 qrels。
- BM25 cache/output：`MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only_audited/`
  - `collection/chunks.jsonl` 约 6.86 MB
  - `chunk_metadata.jsonl` 约 1.66 MB
  - `raw_rankings.jsonl` 约 12.34 MB
  - `index_build.json`、`search_run.json`、`export_report.json`
- 该基础产物的历史全量 test 指标为 Recall@5=0.974、Recall@10=0.984；它不是本次
  MedSearch rollout 的 50 条结果。
- `MedRAG/corpus/` 本地约 663 MB，主要可见 `textbooks/chunk/` 和 `statpearls/chunk/`
  JSONL；MedSearch 的 `medtext_corpus.jsonl` 是已整理的单文件医学语料，不能假设两者
  的 document id/index 可直接互换。

## 4. 可复现命令（CMD 可粘贴）

检查当前 MedSearch 服务是否仍在运行：

```cmd
wsl.exe bash -lc "curl -sS -m 2 -o /tmp/medsearch_retrieve.out -w 'http=%{http_code}\n' -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' --data '{\"queries\":[\"test\"],\"topk\":5,\"return_scores\":true}' || true"
```

确认语料规模和历史服务日志：

```cmd
wsl.exe bash -lc "cd '/mnt/d/code_list/some tricks/LLMLeanring' && wc -l MedSearch/data/stage_b/medtext_corpus.jsonl && du -h MedSearch/data/stage_b/medtext_corpus.jsonl && rg -n 'Started server|Uvicorn|/retrieve.*200' MedSearch/outputs/dapo_epoch1_bm25.log | head -20"
```

复核本次抽样的文件和 gold 数据是否存在：

```cmd
wsl.exe bash -lc "cd '/mnt/d/code_list/some tricks/LLMLeanring' && find MedSearch/outputs/dapo_epoch1_n5_s42/rollouts -maxdepth 1 -name '*.jsonl' | sort -V | head -50 && sha256sum MedSearch/outputs/dapo_epoch1_bm25.log MedSearch/data/stage_b/medtext_corpus.jsonl MedSearch/data/rl/train.parquet MedSearch/data/rl/dev.parquet"
```

复现所需的审计逻辑是：读取 rollout `input` 的 `Question:`，从 train/dev parquet 按 question
匹配 `golden_answers`，读取 `output` 的 `<information>`，解析 `Doc 1..5`，对 gold 做小写、
非字母数字归一化后的子串检查。不要把没有 `<information>` 的 rollout 补成“检索命中”。

## 5. 建议

1. 先恢复 MedSearch BM25 服务和 index，并给服务启动参数写入 manifest：corpus SHA-256、
   index 路径、Pyserini/Java 版本、BM25 k1/b、top-k。
2. 对本次 46 个 strict miss 分层：`no information`、检索到相关主题但没答案、检索到答案
   同义表述但 lexical miss、以及纯错误检索。这样才能区分 retriever error 与 generator error。
3. 对 4 个 strict hit 仍要检查最终答案：若 evidence 已含 gold 但回答错误，更直接支持
   “证据到答案转换/协议/奖励”问题；若 evidence 不含 gold，优先修检索或语料索引。
4. 不要用本次 strict lexical 8% 直接宣称医学证据 Recall；下一轮应保存 `document_id`、
   score、query、gold、answer、evidence span，并增加人工/医学语义支持标签。


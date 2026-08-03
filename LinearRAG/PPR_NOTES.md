# LinearRAG 主干与 PPR 权重说明

本文提炼 LinearRAG 默认非向量化检索主干，以及 Personalized PageRank（PPR）中各类权重的准确含义。对应代码主要位于 `src/LinearRAG.py`。

## 1. LinearRAG 检索主干

```text
Question
→ Question Embedding + Question NER
→ get_seed_entities()
→ Entity → Sentence → Entity 语义传播
→ entity_weights
→ Dense Passage 分数 + 激活 Entity 奖励
→ passage_weights
→ node_weights = entity_weights + passage_weights
→ run_ppr(node_weights)
→ 只保留 Passage 节点
→ 按 PPR 分数降序排列
→ 截取 Top-k Passage
→ 拼接 Prompt
→ LLM 生成答案
→ LLM Accuracy / Contain Accuracy
```

如果 Question NER 没有识别出实体，系统不会运行图检索，而是直接使用 `dense_passage_retrieval()` 返回 Passage。

## 2. PPR 之前计算出的两类节点权重

### 2.1 `entity_weights`

`calculate_entity_scores()` 通过以下路径传播：

```text
Seed Entity
→ 与当前问题最相似的 Sentence
→ Sentence 中的下一层 Entity
```

传播分数为：

```text
下一层 Entity 分数
= 当前 Entity 分数 × Question–Sentence 相似度
```

`entity_weights` 表示当前问题重点关注哪些 Entity 节点。

### 2.2 `passage_weights`

`calculate_passage_scores()` 先计算全部 Passage 与 Question 的 Dense 相似度，再加入已激活 Entity 的出现奖励。

忽略可选属性增强时：

```text
单个 Entity 奖励
= Entity 分数 × log(1 + 出现次数) ÷ tier

Passage 分数
= passage_ratio × 归一化 Dense 分数
 + log(1 + Entity 奖励总和)

Passage 节点权重
= Passage 分数 × passage_node_weight
```

其中：

- Dense 分数回答“这个 Passage 整体语义是否接近问题”；
- Entity 奖励回答“这个 Passage 是否包含实体传播找到的重要实体”；
- `tier` 越大，说明实体距离 Seed 越远，奖励越小。

## 3. `node_weights` 的意义

```python
node_weights = entity_weights + passage_weights
```

两者都与整张 igraph 图的节点位置对齐：

```text
Entity 节点位置  → Entity 权重
Passage 节点位置 → Passage 权重
```

`node_weights` 不是最终检索分数。它是 PPR 的个性化重启偏好，表示：

> 对于当前问题，随机游走重新出发时应该优先回到哪些 Entity 和 Passage。

`run_ppr()` 会先把 `NaN` 和负数裁剪为 0：

```python
reset_prob = np.where(
    np.isnan(node_weights) | (node_weights < 0),
    0,
    node_weights,
)
```

PPR 使用的是这些权重之间的相对比例。例如 `[8, 2, 0]` 与 `[0.8, 0.2, 0]` 表示相同的重启偏好。

## 4. 图的边传递什么

边上传递的不是原始 `node_weights`，也不是边权本身，而是：

> 当前一轮中，节点持有的 PPR 概率质量。

假设 Entity E 当前分数为 `0.8`，并有两条边：

```text
E ——权重 3—— P1
E ——权重 1—— P2
```

E 的当前分数按照相邻边权归一化分配：

```text
传给 P1：0.8 × 3/(3+1) = 0.6
传给 P2：0.8 × 1/(3+1) = 0.2
```

因此，Edge weight 的作用是决定：

> 当前节点的 PPR 分数按照什么比例分给相邻节点。

## 5. “按重启权重重新注入”是什么意思

假设归一化后的重启分布为：

```text
r = [Entity E: 0.8, Passage P1: 0.2, Passage P2: 0]
```

并且：

```text
damping = 0.5
```

那么每轮固定加入的重启部分为：

```text
(1 - damping) × r
= 0.5 × [0.8, 0.2, 0]
= [0.4, 0.1, 0]
```

这表示每轮都把一部分概率重新拉回与当前问题相关的节点，防止概率完全扩散到无关区域。

## 6. PPR 每轮的更新公式

```text
下一轮 PPR 分数
= damping × 沿加权边传播的结果
 + (1 - damping) × 重启分布
```

常写为：

```text
p(t+1) = damping × Pᵀp(t) + (1-damping) × r
```

其中：

- `p(t)`：当前一轮所有节点的 PPR 分数；
- `P`：由图连接和归一化边权形成的转移概率；
- `r`：由 `node_weights` 形成的个性化重启分布。

igraph 会反复计算，直到节点分数稳定。稳定后的 `pagerank_scores` 表示：

> 在“按当前问题重启，并沿加权图游走”的规则下，长期访问每个节点的概率。

## 7. `personalized_pagerank()` 参数

```python
pagerank_scores = self.graph.personalized_pagerank(
    vertices=range(len(self.node_name_to_vertex_idx)),
    damping=self.config.damping,
    directed=False,
    weights="weight",
    reset=reset_prob,
    implementation="prpack",
)
```

| 参数 | 含义 |
|---|---|
| `vertices=...` | 为全部图节点返回 PPR 分数，包括 Entity 和 Passage |
| `damping` | 沿边传播的比例；`1-damping` 是按重启分布重新出发的比例 |
| `directed=False` | 把图作为无向图，分数可以沿边双向传播 |
| `weights="weight"` | 使用边的 `weight` 属性决定相邻节点之间的传播比例 |
| `reset=reset_prob` | 使用当前问题对应的 Entity/Passage 节点权重作为重启偏好 |
| `implementation="prpack"` | 使用 PRPACK 数值后端求解，不改变 PPR 的语义 |

## 8. Node、Edge 与 damping 的职责

| 信息 | 解决的问题 |
|---|---|
| Node weight | 当前问题重点关注哪些 Entity 和 Passage？ |
| Edge weight | 当前节点的分数应该按什么比例传给哪些邻居？ |
| `damping` | 每轮有多少分数沿边传播，有多少分数按问题相关节点重启？ |
| PPR score | 多轮传播并稳定后，每个节点长期占有多少概率？ |

最简记忆：

```text
Node weight：每轮优先回到哪里
Edge weight：当前分数怎样分给邻居
damping：沿边走与重新出发各占多少
PPR score：反复执行后每个节点稳定占多少概率
```

## 9. PPR 之后的处理

PPR 会为 Entity 和 Passage 全部打分，但 Entity 只是传播中间节点，不能直接作为完整证据放入生成模型。因此代码只保留 Passage 节点：

```text
全部节点的 PPR 分数
→ 按 passage_node_indices 取出 Passage 分数
→ 降序排列
→ Passage 节点位置转 hash ID
→ retrieve() 截取 retrieval_top_k
→ hash ID 转 Passage 文本
```

随后 `qa()` 把 Top-k Passage 拼进 Prompt，调用 LLM 生成答案。`Evaluator` 计算的是最终答案的 LLM Accuracy 和 Contain Accuracy，不是 Recall@k、MRR 或 nDCG。

## 10. 常见误解

1. **边权不是直接加到 Passage 分数上。** 它控制当前 PPR 分数在邻居之间的分配比例。
2. **`node_weights` 不是最终排名。** 它是每轮重启时的问题相关偏好。
3. **沿边传播的不是固定初始权重。** 传播的是上一轮不断变化的 PPR 概率质量。
4. **PPR 分数不等于 Dense 分数。** 它综合了节点重启偏好、图结构、边权和多轮传播。
5. **PPR 输出全部节点。** LinearRAG 最终只筛选 Passage 节点作为检索证据。

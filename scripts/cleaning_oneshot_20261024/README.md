# 清洗批次一次性脚本（2026-10-24 ~ 2026-10-26）

这些是 Luna Max 清洗批次执行时用的一次性写盘/核对脚本，原始位置在仓库根目录，
2026-11-02 大扫除时按「整理到一个文件夹、不删除」的要求集中到这里，共 45 个文件、约 0.21 MB。

## 覆盖的批次

| 批次目录 | 脚本数 |
|---|---:|
| `MedSearch/data/cleaning_v2/rl_20261024_lunamax_batch08` | 12 |
| `MedSearch/data/cleaning_v2/rl_20261024_lunamax_batch11` | 14 |
| `MedSearch/data/cleaning_v2/rl_20261024_lunamax_batch12` | 14 |
| `MedSearch/data/cleaning_v2/rl_20261026_lunamax_batch26` | 2 |

## 脚本分三类

- `_build_*`：把审阅结论写回 `review/raw_*.jsonl`，或生成 `review/input_*.jsonl`（C1 语义对 / Q2 答案覆盖 / C3 证据窗口）。
- `_inspect_*`、`_show_*`、`_print_*`、`_passages_*`、`_status_*`、`_summarize_*`、`_tail_*`、`_check_rollout_*`：只读查看与核对，不改数据。
- `.tmp_read_c3_batch26_02.py` / `.tmp_write_c3_batch26_02.py`：batch26 第 2 个分片的读写。

## 怎么运行

脚本内部全部使用 `Path("MedSearch/data/cleaning_v2/...")` 或 `Path.cwd()`，**没有** 用 `__file__` 定位，
所以必须在仓库根目录执行：

```powershell
cd "D:\code_list\some tricks\LLMLeanring"
python scripts\cleaning_oneshot_20261024\_build_c3_12_01.py
```

在别的目录执行会找不到 `MedSearch/data/...`。

## 注意

- 这些脚本是逐批一次性的，同一批次跑完后一般不重复执行；重跑会覆盖对应 `review/raw_*.jsonl`。
- 它们只写 review 分片账本，不改 `MedSearch/data/cleaning_v2` 里的发布文件、`CURRENT_*` 指针或 gold。
- 本次搬动只改位置，未改任何一行内容。记录见仓库根 `CLEANUP_20261102.md`。

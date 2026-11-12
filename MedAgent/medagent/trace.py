"""Public audit-trace helpers for MedAgent."""


# ============================================================================
# P2-c：公共 Trace 脱敏
#
# redact() 负责把内部 payload 投影为允许公开的顶层字段。
# 它返回新字典，不修改输入对象。
#
# 只保留 allowed_fields 中实际存在的字段；未列入白名单的密钥、身份
# 信息、原始医学文本和隐藏推理不会被保留。
#
# 本阶段不做递归脱敏、敏感词匹配、医学文本识别或文件写入。
# ============================================================================


def redact(
    payload: dict[str, object],
    allowed_fields: tuple[str, ...],
) -> dict[str, object]:
    """
    按白名单提取可公开的顶层字段，生成脱敏后的新字典。

    Input:
        payload (dict[str, object]): 内部待投影的对象。
        allowed_fields (tuple[str, ...]): 允许公开的顶层字段名元组。
    Output:
        dict[str, object]: 仅包含白名单字段的新字典，不修改 payload。
    """

    return {
        field: payload[field]
        for field in allowed_fields
        if field in payload
    }


import json
from pathlib import Path
from .contracts import TraceEvent, ReplayResult
from dataclasses import asdict

class TraceStore:
    """管理一次运行的 manifest、公开事件和最终结果文件。"""

    def __init__(self, root: Path):
        """
        初始化 TraceStore，并准备运行目录和事件序号记录。

        Input:
            root (Path): Trace 运行目录的根路径。
        Output:
            None：初始化目录、当前运行路径和事件序号集合。
        """
        self.root = root
        self.run_dir: Path | None = None                # run_id存放位置
        self.sequences: set[str] = set()                # sequences与run_id配套锚定唯一事件

    def start(self, manifest: dict[str, object]) -> None:
        """
        创建本次运行目录，并将 manifest 保存为 manifest.json。

        Input:
            manifest (dict[str, object]): 运行清单，必须包含非空 run_id。
        Output:
            None：创建 manifest.json，并设置当前运行目录。
        """

        # region 运行状态与 manifest 校验
        if self.run_dir is not None:
            raise RuntimeError("trace store already started")
        if not isinstance(manifest, dict):
            raise TypeError("manifest must be a dictionary")
        if "run_id" not in manifest:
            raise ValueError("manifest requires run_id")

        run_id = manifest["run_id"]
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        # endregion

        # 序列化, 创建目录+json

        text = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        )

        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(text, encoding="utf-8")
        # 设置run_dir目录
        self.run_dir = run_dir

    def append(self, event: TraceEvent) -> None:
        """
        校验并追加一条公开 Trace 事件到 events.jsonl。

        Input:
            event (TraceEvent): 需要追加的公开 Trace 事件。
        Output:
            None：将事件写入 events.jsonl，并记录其 sequence。
        """

        # region 运行状态与 TraceEvent 校验
        # 判断start()是否执行
        if self.run_dir is None:
            raise RuntimeError("trace store has not started")
        # event类型错误
        if not isinstance(event, TraceEvent):
            raise TypeError("event must be a TraceEvent")

        # event.run_id 与当前目录不一致
        if event.run_id != self.run_dir.name:
            raise ValueError("event run_id does not match current run")

        # sequence已写入
        if event.sequence in self.sequences:
            raise ValueError("duplicate event sequence")
        # endregion

        # event类转化为dict后保存为json
        payload = asdict(event)
        line = json.dumps(payload, ensure_ascii=False)

        # event结果写入
        events_path = self.run_dir / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as f:
            f.write(line + '\n')
        # 追加到sequences
        self.sequences.add(event.sequence)

    def finish(self, result: dict[str, object]) -> None:
        """
        保存本次运行的最终答案和运行摘要。

        Input:
            result (dict[str, object]): 包含 answer 和 summary 的结果对象。
                answer: 结构化最终答案字典。
                summary: 结构化运行摘要字典。
        Output:
            None：将 answer 和 summary 分别保存为 JSON 文件。
        """
        # region 运行状态与结果校验
        # 判断start()是否执行
        if self.run_dir is None:
            raise RuntimeError("trace store has not started")

        # result必须为dict
        if not isinstance(result, dict):
            raise TypeError("result must be a dictionary")

        # result必须包含answer和summary
        if not "answer" in result or not "summary" in result:
            raise ValueError("result requires answer and summary")

        # answer/summary必须字典且表示结构化内容
        if not isinstance(result["answer"], dict) or not isinstance(result["summary"], dict):
            raise TypeError("answer and summary must be dictionaries")
        # endregion

        # 序列化表示内容
        answer_text = json.dumps(
            result["answer"],
            ensure_ascii=False,
            indent=2,
        )
        summary_text = json.dumps(
            result["summary"],
            ensure_ascii=False,
            indent=2,
        )

        answer_path = self.run_dir / "answer.json"
        with answer_path.open("x", encoding="utf-8") as f:
            f.write(answer_text)

        summary_path = self.run_dir / "summary.json"
        with summary_path.open("x", encoding="utf-8") as f:
            f.write(summary_text)

def replay(path: Path) -> ReplayResult:

    # region 路径校验
    # 确认运行目录
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    # endregion

    # 读取manifest.json
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # region manifest.json 校验
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dictionary")
    
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest requires a non-empty run_id")
    # endregion

    # 读取events.jsonl
    events_path = path / "events.jsonl"

    if not events_path.exists():
        events = ()
    else:
        events = tuple(
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    # region events.jsonl 校验
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("events must contain dictionaries")
    # endregion

    # 读取summary和answer
    answer_path = path / "answer.json"

    if not answer_path.exists():
        answer = None
    else:
        answer = json.loads(answer_path.read_text(encoding="utf-8"))


    summary_path = path / "summary.json"
    if not summary_path.exists():
        summary = None
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # region answer 和 summary 完整性校验
    if (answer is None) != (summary is None):
        raise ValueError("incomplete final artifacts")
    # endregion

    result = ReplayResult(
        manifest=manifest,
        events=events,
        answer=answer,
        summary=summary,
    )

    return result

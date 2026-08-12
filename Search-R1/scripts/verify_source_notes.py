"""验证中文学习注释没有改变官方 Python/Bash 执行逻辑。

在仓库根目录运行：
    python Search-R1/scripts/verify_source_notes.py

Python 比较去除模块/类/函数 docstring 后的 AST；Bash 比较去除注释和空行后的
有效命令行。基线固定为本项目引入官方快照的 commit ``3d4832d``，所以提交或 pull
之后仍可复验。脚本只用标准库，不导入 Search-R1 的重型训练依赖。
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Any, List, Tuple


PYTHON_FILES = (
    "Search-R1/scripts/data_process/nq_search.py",
    "Search-R1/search_r1/llm_agent/generation.py",
    "Search-R1/search_r1/llm_agent/tensor_helper.py",
    "Search-R1/search_r1/search/retrieval_server.py",
    "Search-R1/verl/utils/dataset/rl_dataset.py",
    "Search-R1/verl/utils/reward_score/qa_em.py",
    "Search-R1/verl/trainer/main_ppo.py",
    "Search-R1/verl/trainer/ppo/ray_trainer.py",
)
BASH_FILES = ("Search-R1/train_grpo.sh",)
OFFICIAL_BASELINE = "3d4832d"


def read_baseline(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{OFFICIAL_BASELINE}:{path}"], text=True, encoding="utf-8"
    )


def ast_without_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    return ast.dump(tree, include_attributes=False)


def bash_commands(source: str) -> list[str]:
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def verify_reward_examples() -> None:
    """直接加载纯标准库 reward 文件，避免导入 veRL 的训练依赖。"""
    path = "Search-R1/verl/utils/reward_score/qa_em.py"
    spec = importlib.util.spec_from_file_location("qa_em_source_notes_check", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    gold = {"target": ["Beijing"]}
    assert module.compute_score_em(
        "<answer>example</answer><answer>Beijing</answer>", gold
    ) == 1
    assert module.compute_score_em("<answer>Beijing</answer>", gold) == 0
    assert module.compute_score_em(
        "<answer>example</answer><answer>Shanghai</answer>", gold
    ) == 0


def verify_action_parser_examples() -> None:
    """只执行源码中的 parser 方法，绕开尚未安装的 veRL/tensordict。"""
    tree = ast.parse(
        Path("Search-R1/search_r1/llm_agent/generation.py").read_text(encoding="utf-8")
    )
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMGenerationManager"
    )
    parser = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "postprocess_predictions"
    )
    namespace = {"re": re, "List": List, "Any": Any, "Tuple": Tuple}
    exec(compile(ast.fix_missing_locations(ast.Module([parser], [])), "generation.py", "exec"), namespace)
    actions, contents = namespace["postprocess_predictions"](
        object(), ["<search>aspirin dose</search>", "x<answer>A</answer>", "plain"]
    )
    assert actions == ["search", "answer", None]
    assert contents == ["aspirin dose", "A", ""]


def main() -> None:
    changed = []
    for path in PYTHON_FILES:
        before = ast_without_docstrings(read_baseline(path))
        after = ast_without_docstrings(Path(path).read_text(encoding="utf-8"))
        if before != after:
            changed.append(path)

    for path in BASH_FILES:
        before = bash_commands(read_baseline(path))
        after = bash_commands(Path(path).read_text(encoding="utf-8"))
        if before != after:
            changed.append(path)

    if changed:
        raise SystemExit("Executable behavior changed: " + ", ".join(changed))
    verify_reward_examples()
    verify_action_parser_examples()
    print(f"Comment-only verification passed: {len(PYTHON_FILES)} Python + {len(BASH_FILES)} Bash files")
    print("Behavior checks passed: 3 reward + 3 action parser examples")


if __name__ == "__main__":
    main()

#!/bin/bash
# 补齐剩余实验:scifact graph-sim/hybrid + pubmedqa reranker 三变体
set -e
cd "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG"
GIT=$(git rev-parse HEAD)
IMG=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
mkdir -p logs

/opt/venv/bin/python -m medical_graphrag.cli run graph-document --dataset scifact_v1 --profile similarity --git-commit "$GIT" --docker-image "$IMG" > logs/p2_scifact.log 2>&1
echo "scifact graph-sim exit=$?"

/opt/venv/bin/python -m medical_graphrag.cli run hybrid-document --dataset scifact_v1 --git-commit "$GIT" --docker-image "$IMG" >> logs/p2_scifact.log 2>&1
echo "scifact hybrid exit=$?"

for s in bd bdg bde; do
  /opt/venv/bin/python -m medical_graphrag.cli run reranker-document --dataset pubmedqa_hard_v1 --sources "$s" --top-n 30 --git-commit "$GIT" --docker-image "$IMG" > logs/reranker_$s.log 2>&1
  echo "reranker-$s exit=$?"
done

echo "ALL DONE"

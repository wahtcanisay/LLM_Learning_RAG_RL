# MedicalGraphRAG

Standalone implementation for reproducible medical retrieval experiments.

The sibling `MedRAG/` and `LinearRAG/` directories are read-only references and
data sources. This project owns all new cleaning, retrieval, evaluation, tests,
and experiment records.

The first milestone is `pubmedqa_hard_v1`: 1,000 PubMedQA gold documents plus
4,000 deterministic MedRAG PubMed distractors with document-level qrels.

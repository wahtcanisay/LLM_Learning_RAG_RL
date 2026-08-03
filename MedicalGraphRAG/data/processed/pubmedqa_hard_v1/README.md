# pubmedqa_hard_v1

Expected composition after the full build:

- 1,000 PubMedQA questions and gold documents;
- 4,000 deterministic MedRAG PubMed distractors;
- exactly one document-level qrel per question;
- chunks that never cross source-document or PubMedQA context boundaries.

The 20-question audit must pass before the full artifacts are generated.

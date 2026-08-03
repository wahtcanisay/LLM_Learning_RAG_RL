# Raw data policy

Raw PubMedQA files are downloaded from the official repository:

- `https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json`
- `https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json`

The files are ignored by Git. Their SHA-256 values are written to the generated
benchmark manifest so a run can be reproduced and audited.

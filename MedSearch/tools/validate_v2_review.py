import collections
import hashlib
import json
from pathlib import Path

root = Path('.')
p = root / 'MedSearch/data/medical_verifiable_v2_20260919'
def read(path):
    return [json.loads(x) for x in path.open(encoding='utf-8') if x.strip()]
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

raw_path = root / 'MedSearch/data/raw/medical_o1_verifiable_problem.json'
raw = json.loads(raw_path.read_text(encoding='utf-8'))
candidates = read(p / 'candidate_keep.jsonl')
ledger = read(p / 'review_ledger.jsonl')
queue = read(p / 'semantic_neighbors.jsonl')
pairs = read(p / 'semantic_pair_review.jsonl')
lexical = read(p / 'lexical_neighbors.jsonl')
lexical_pairs = read(p / 'lexical_pair_review.jsonl')
manual = read(p / 'manual_review_notes.jsonl')
cm = json.loads((p / 'manifest.json').read_text(encoding='utf-8'))
lm = json.loads((p / 'review_ledger.manifest.json').read_text(encoding='utf-8'))
pm = json.loads((p / 'semantic_pair_review.manifest.json').read_text(encoding='utf-8'))
lexm = json.loads((p / 'lexical_neighbors.manifest.json').read_text(encoding='utf-8'))
lexpm = json.loads((p / 'lexical_pair_review.manifest.json').read_text(encoding='utf-8'))
exact = json.loads((root / 'MedSearch/docs/audits/distillation_v2_candidate_isolation_container.json').read_text(encoding='utf-8'))
assert len(raw) == 40644 and len(candidates) == len(ledger) == 3000
assert sha(raw_path) == cm['source_sha256']
assert sha(p / 'candidate_keep.jsonl') == cm['outputs_sha256']['candidate_keep.jsonl'] == lm['candidate_sha256'] == exact['inputs']['v2']['sha256']
assert sha(p / 'semantic_neighbors.jsonl') == lm['neighbor_sha256'] == pm['semantic_queue_sha256']
assert sha(p / 'lexical_neighbors.jsonl') == lm['lexical_neighbor_sha256'] == lexm['output_sha256'] == lexpm['lexical_queue_sha256']
assert sha(p / 'review_ledger.jsonl') == lm['ledger_sha256']
assert sha(p / 'semantic_pair_review.jsonl') == pm['pair_review_sha256']
assert sha(p / 'lexical_pair_review.jsonl') == lexpm['new_pair_review_sha256']
assert len(manual) == len({x['question_id'] for x in manual}) == lm['manual_review_rows']
for candidate, reviewed in zip(candidates, ledger, strict=True):
    source = raw[candidate['source_id']]
    assert candidate['question'] == source['Open-ended Verifiable Question']
    assert candidate['references'] == [source['Ground-True Answer']]
    assert all(candidate[k] == reviewed[k] for k in candidate)
assert len(queue) == len(pairs) == 304
assert len(lexical) == 184 and len(lexical_pairs) == 146
for queued, reviewed in zip(queue, pairs, strict=True):
    assert all(queued[k] == reviewed[k] for k in ('kind', 'candidate', 'match', 'matched_pool', 'matched_row', 'cosine'))
assert collections.Counter(row['pair_verdict'] for row in pairs) == {'same_question': 281, 'uncertain': 23}
assert collections.Counter(row['decision'] for row in ledger) == {'reject': 294, 'quarantine': 120, 'pending': 2586}
assert sum(row['matched_pool'] == 'final_test' and row['pair_verdict'] == 'same_question' for row in pairs) == 15
assert exact['conflicts'] == [] and exact['missing'] == [] and exact['exact_check_passed']
assert not lm['training_ready'] and lm['accepted_rows'] == 0 and lm['split_assignment'] == 'not_done'
print('validated: raw gold/question preservation 3000; semantic pairs 304; lexical queue 184; exact conflicts 0; decision counts 294/120/2586; training_ready false')

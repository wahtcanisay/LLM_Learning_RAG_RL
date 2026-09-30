import json
import hashlib
from pathlib import Path

d = Path('MedSearch/data/medical_verifiable_v2_batch2_20260926')
p = d / 'manual_review_notes.jsonl'
notes = [json.loads(line) for line in p.read_text(encoding='utf8').splitlines()]
existing = {row['question_id'] for row in notes}
new = {
    12119: ('quarantine', 'critical_question_context_missing', 'Question invokes CT findings but gives no actual CT features; clipping is only one possible aneurysmal SAH intervention.'),
    12808: ('quarantine', 'underspecified_question', 'Fatigue, fever, anorexia and weight loss alone do not uniquely support the gold leukemia diagnosis.'),
    13909: ('quarantine', 'underspecified_question', 'Rough-ER-bound ribosomes make several protein classes; question provides no qualifier selecting lysosomes uniquely.'),
    14725: ('quarantine', 'critical_question_context_missing', 'Question asks about over-replenishment of an implicated serum electrolyte, but no electrolyte or serum value is given.'),
    18077: ('quarantine', 'critical_question_context_missing', 'Question omits the newly started drug and claimed current medication regimen; gold only says generic adverse medication effect.'),
    19058: ('reject', 'unsupported_gold_and_harmful_inference', 'Question supplies no evidence identifying a perpetrator; labeling the mother as abuser is unsupported by the stated facts.'),
    19255: ('quarantine', 'critical_question_context_missing', 'Question relies on MRI findings but provides no MRI description; rapid decline alone does not establish CJD.'),
    23705: ('quarantine', 'critical_question_context_missing', 'Question says medications were specified but omits the medication list needed to choose furosemide.'),
}
for source_id, (decision, reason, evidence) in new.items():
    question_id = f'medical-o1-verifiable:{source_id}'
    if question_id not in existing:
        notes.append({'question_id':question_id,'decision':decision,'reason':reason,'reviewer':'Luna','review_basis':'manual_question_gold','evidence':evidence})
notes.sort(key=lambda row: int(row['question_id'].split(':')[-1]))
p.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in notes), encoding='utf8', newline='\n')
pair_manifest_path = d / 'pair_review.manifest.json'
pair_manifest = json.loads(pair_manifest_path.read_text(encoding='utf8'))
with p.open('rb') as handle:
    pair_manifest['manual_notes_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
pair_manifest['manual_candidate_rows'] = len(notes)
pair_manifest['additional_question_gold_quality_notes'] = len(new)
pair_manifest_path.write_text(json.dumps(pair_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print({'new_quality_notes': len(new), 'total_manual_notes': len(notes), 'overlaps_with_prior_notes': len(new) + len(existing) - len(notes)})

import collections
import hashlib
import json
from pathlib import Path

p=Path('MedSearch/data/medical_verifiable_v2_batch2_20260926')
def read(name):
    return [json.loads(x) for x in (p/name).open(encoding='utf-8')]
def write(name,rows):
    with (p/name).open('w',encoding='utf-8',newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()
semantic=read('semantic_neighbors.jsonl')
lexical=read('lexical_neighbors.jsonl')
assert len(semantic)==82 and len(lexical)==50
exceptions={
    7:'Patient sex and geographic context differ (Bihar woman versus Jaipur man).',
    18:'Patient age differs three versus six years in a pediatric case.',
    19:'Vocabulary size differs 4–6 versus 10–15 words; developmental age may differ.',
    49:'Hepatorenomegaly versus hepatosplenomegaly may change diagnostic evidence.',
    71:'Most common congenital anomaly versus one common anomaly has different answer scope.',
    81:'Intravenous induction alone versus induction and maintenance changes the agent requirement.',
}
notes={}
pair_reviews=[]
def add_note(item_id,decision,reason,evidence):
    current=notes.get(item_id)
    if current and current['decision']=='reject':
        return
    if current and current['decision']==decision:
        return
    notes[item_id]={
        'question_id':item_id,'decision':decision,'reason':reason,'reviewer':'Luna',
        'review_basis':'manual_pair_text','evidence':evidence,
    }
for index,row in enumerate(semantic):
    uncertain=index in exceptions
    reason=exceptions[index] if uncertain else 'Same core question and material facts despite paraphrase.'
    pair_reviews.append({
        'queue_index':index,'kind':row['kind'],'candidate':row['candidate'],'match':row['match'],
        'matched_pool':row['matched_pool'],'matched_row':row['matched_row'],'cosine':row['cosine'],
        'pair_verdict':'uncertain' if uncertain else 'same_question',
        'basis':'Luna_pair_text_review_not_clinical_gold_review','reason':reason,
    })
    locator=f"{row['matched_pool']} row {row['matched_row']} / {row['match']}"
    if uncertain:
        add_note(row['candidate'],'quarantine','similarity_not_confirmed_duplicate',f'Read pair with {locator}: {reason} Medical gold not reviewed.')
        if row['kind']=='internal':
            add_note(row['match'],'quarantine','similarity_not_confirmed_duplicate',f'Read pair with {row["candidate"]}: {reason} Medical gold not reviewed.')
    elif row['kind']=='history':
        add_note(row['candidate'],'reject','same_question_as_frozen_test' if row['matched_pool']=='final_test' else 'same_question_as_historical_or_batch1',f'Read pair with {locator}: {reason} Medical gold not reviewed.')
    else:
        reject_id=max((row['candidate'],row['match']),key=lambda value:int(value.split(':')[-1]))
        add_note(reject_id,'reject','same_question_as_internal_batch2',f'Read internal pair {row["candidate"]} / {row["match"]}: {reason} Medical gold not reviewed.')
write('semantic_pair_review.jsonl',pair_reviews)

def key(row):
    return (row['kind'],row['candidate'],row['match']) if row['kind']=='history' else ('internal',*sorted((row['candidate'],row['match'])))
semkeys={key(x) for x in semantic}
semids={x['candidate'] for x in semantic}|{x['match'] for x in semantic if x['kind']=='internal'}
lex_new=[]
for index,row in enumerate(lexical):
    if key(row) in semkeys:
        continue
    if row['candidate'] in semids:
        verdict='not_separately_adjudicated'
        reason='Candidate already represented in semantic queue; this additional lexical relation was not separately adjudicated.'
        basis='already_in_semantic_queue_not_pair_reviewed'
    else:
        verdict='same_question'
        reason='Same core question and material facts despite paraphrase.'
        basis='Luna_pair_text_review_not_clinical_gold_review'
        add_note(row['candidate'],'reject','same_question_as_historical_or_batch1',f'Read lexical pair with {row["matched_pool"]} row {row["matched_row"]} / {row["match"]}: {reason} Medical gold not reviewed.')
    lex_new.append({
        'queue_index':index,'kind':row['kind'],'candidate':row['candidate'],'match':row['match'],
        'matched_pool':row['matched_pool'],'matched_row':row['matched_row'],'cosine':row['cosine'],
        'pair_verdict':verdict,'basis':basis,'reason':reason,
    })
assert len(lex_new)==44
assert collections.Counter(x['pair_verdict'] for x in lex_new)=={'not_separately_adjudicated':40,'same_question':4}
write('lexical_pair_review.jsonl',lex_new)
write('manual_review_notes.jsonl',sorted(notes.values(),key=lambda x:int(x['question_id'].split(':')[-1])))
report={
    'status':'pair_text_review_only_medical_gold_not_reviewed',
    'semantic_queue_sha256':sha(p/'semantic_neighbors.jsonl'),
    'lexical_queue_sha256':sha(p/'lexical_neighbors.jsonl'),
    'semantic_pair_review_sha256':sha(p/'semantic_pair_review.jsonl'),
    'lexical_pair_review_sha256':sha(p/'lexical_pair_review.jsonl'),
    'manual_notes_sha256':sha(p/'manual_review_notes.jsonl'),
    'semantic_pair_verdicts':dict(collections.Counter(x['pair_verdict'] for x in pair_reviews)),
    'lexical_new_relation_verdicts':dict(collections.Counter(x['pair_verdict'] for x in lex_new)),
    'manual_candidate_rows':len(notes),
    'medical_review':'not_done',
}
(p/'pair_review.manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(report)

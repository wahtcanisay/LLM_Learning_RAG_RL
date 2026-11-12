"""Derive the CE control by dropping only teacher probability cache columns."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    source = Path('MedSearch/data/distillation/top32_sft1000_20261011/train.parquet')
    root = Path('MedSearch/data/distillation/ce_sft1000_20261013')
    if root.exists():
        raise FileExistsError(root)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == '83dd481cbb2f248b407490367d787a42f0dbde021b70ddacd647c717ed45c4e5'
    table = pq.read_table(source)
    cache_columns = ['target_positions', 'teacher_topk_ids', 'teacher_topk_logprobs']
    assert table.num_rows == 1000 and all(name in table.column_names for name in cache_columns)
    ce = table.drop(cache_columns)
    root.mkdir(parents=True)
    path = root / 'train.parquet'
    pq.write_table(ce, path)
    actual = pq.read_table(path)
    assert actual.equals(ce)  # 所有非缓存列（含ID、顺序、token和mask）逐值相同。
    assert not any(name in actual.column_names for name in cache_columns)
    report = {'status': 'passed', 'rows': actual.num_rows, 'source': str(source),
              'source_sha256': digest, 'ce_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'dropped_columns': cache_columns, 'all_other_columns_identical': True}
    (root / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

from MedSearch.rl.distillation_hook import build_teacher_score_jobs


def test_teacher_jobs_keep_observations_in_prefix_only():
    jobs = build_teacher_score_jobs(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 90, 91, 30],
        response_mask=[1, 1, 0, 0, 1],
    )
    assert jobs == [
        {"prefix_ids": [10, 11], "target_ids": [20, 21], "response_start": 0},
        {"prefix_ids": [10, 11, 20, 21, 90, 91], "target_ids": [30], "response_start": 4},
    ]


def test_teacher_jobs_ignore_observation_only_trace():
    assert build_teacher_score_jobs([10], [90, 91], [0, 0]) == []

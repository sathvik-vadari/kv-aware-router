"""The workload has to contain the reuse structure policies are judged on."""

from kv_aware_router.workload import WorkloadSpec, generate, reuse_ceiling

SPEC = WorkloadSpec(n_sessions=30, turns_per_session=4, n_system_prompts=2, seed=7)


def test_requests_are_arrival_ordered():
    reqs = generate(SPEC)
    assert [r.arrival_s for r in reqs] == sorted(r.arrival_s for r in reqs)


def test_turns_within_a_session_stay_in_order():
    reqs = generate(SPEC)
    for sid in {r.session_id for r in reqs}:
        turns = [r.turn for r in reqs if r.session_id == sid]
        assert turns == sorted(turns)


def test_sessions_interleave():
    # if each session ran to completion before the next began, routing would be
    # trivial and every policy would score the same
    reqs = generate(SPEC)
    switches = sum(
        1 for a, b in zip(reqs, reqs[1:]) if a.session_id != b.session_id
    )
    assert switches > len(reqs) // 2


def test_prompt_grows_with_each_turn():
    reqs = generate(SPEC)
    for sid in {r.session_id for r in reqs}:
        sizes = [r.n_tokens for r in sorted(
            (r for r in reqs if r.session_id == sid), key=lambda r: r.turn
        )]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]


def test_unrelated_sessions_share_a_system_prompt_head():
    reqs = generate(WorkloadSpec(n_sessions=20, n_system_prompts=1, seed=1))
    first_turns = [r for r in reqs if r.turn == 0]
    a, b = first_turns[0], first_turns[1]
    assert a.session_id != b.session_id
    assert a.tokens[:512] == b.tokens[:512]


def test_reuse_ceiling_is_a_fraction():
    assert 0.0 < reuse_ceiling(generate(SPEC)) < 1.0

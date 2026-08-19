"""Semantics tests. Each one pins down a rule that is easy to get subtly wrong."""

from kv_aware_router.radix import PrefixTree

SEQ = tuple(range(200))


def test_match_rounds_down_to_block_boundary():
    # a replica holding 100 tokens can only serve 96 of them: the trailing
    # partial block has to be re-prefilled
    tree = PrefixTree(block_size=16)
    tree.insert(SEQ[:100], "a", now=1.0)
    assert tree.match(SEQ[:100]) == {"a": 96}


def test_shorter_query_matches_longer_cached_prefix():
    # residency is inherited upward: holding 200 tokens means holding the first 50
    tree = PrefixTree(block_size=16)
    tree.insert(SEQ, "a", now=1.0)
    assert tree.match(SEQ[:50]) == {"a": 48}


def test_divergent_sequences_share_their_common_prefix():
    tree = PrefixTree(block_size=16)
    tree.insert(SEQ[:100], "a", now=1.0)
    tree.insert(SEQ[:50] + tuple(range(1000, 1050)), "b", now=2.0)
    # querying a's full sequence: a serves 96, b serves only the shared 50 -> 48
    assert tree.match(SEQ[:100]) == {"a": 96, "b": 48}


def test_no_match_when_nothing_shared():
    tree = PrefixTree(block_size=16)
    tree.insert(SEQ[:100], "a", now=1.0)
    assert tree.match(tuple(range(5000, 5100))) == {}


def test_match_below_one_block_is_not_a_hit():
    # 10 shared tokens with a 16-token block size is worth nothing
    tree = PrefixTree(block_size=16)
    tree.insert(SEQ[:10], "a", now=1.0)
    assert tree.match(SEQ[:10]) == {}


def test_eviction_is_lru_across_independent_prefixes():
    tree = PrefixTree(block_size=16, capacity_tokens={"a": 100})
    tree.insert(tuple(range(0, 50)), "a", now=1.0)
    tree.insert(tuple(range(100, 150)), "a", now=2.0)
    tree.insert(tuple(range(200, 250)), "a", now=3.0)   # overflows, evicts oldest
    assert tree.match(tuple(range(0, 50))) == {}         # t=1 gone
    assert tree.match(tuple(range(100, 150))) == {"a": 48}
    assert tree.match(tuple(range(200, 250))) == {"a": 48}
    assert tree.tokens_used["a"] == 100


def test_eviction_takes_the_tail_and_degrades_gracefully():
    # a prefix block cannot be freed while a longer prefix extends it, so the
    # deep suffix goes first and the shorter prefix survives as a partial hit
    tree = PrefixTree(block_size=16, capacity_tokens={"a": 250})
    tree.insert(SEQ[:100], "a", now=1.0)
    tree.insert(SEQ[:200], "a", now=2.0)
    tree.insert(tuple(range(300, 400)), "a", now=3.0)    # overflows
    assert tree.match(SEQ[:200]) == {"a": 96}            # deep half evicted
    assert tree.match(SEQ[:100]) == {"a": 96}            # shallow half survived
    assert tree.evictions["a"] == 1


def test_touch_refreshes_lru_and_changes_the_victim():
    tree = PrefixTree(block_size=16, capacity_tokens={"a": 100})
    tree.insert(tuple(range(0, 50)), "a", now=1.0)
    tree.insert(tuple(range(100, 150)), "a", now=2.0)
    tree.touch(tuple(range(0, 50)), "a", now=3.0)        # oldest becomes newest
    tree.insert(tuple(range(200, 250)), "a", now=4.0)
    assert tree.match(tuple(range(0, 50))) == {"a": 48}  # survived
    assert tree.match(tuple(range(100, 150))) == {}      # evicted instead


def test_remove_replica_clears_it_from_every_node():
    tree = PrefixTree(block_size=16)
    tree.insert(SEQ[:100], "a", now=1.0)
    tree.insert(SEQ[:100], "b", now=1.0)
    tree.remove_replica("a")
    assert tree.match(SEQ[:100]) == {"b": 96}
    assert "a" not in tree.tokens_used


def test_insert_is_idempotent_for_the_same_replica():
    tree = PrefixTree(block_size=16)
    tree.insert(SEQ[:100], "a", now=1.0)
    tree.insert(SEQ[:100], "a", now=2.0)
    assert tree.tokens_used["a"] == 100

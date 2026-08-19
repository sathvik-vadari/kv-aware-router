"""A radix tree over token sequences, tracking which replicas hold which prefix.

This is the router's model of the fleet's collective KV cache. It answers one
question: *given this request's tokens, which replica already holds the longest
usable prefix?*

Design notes, because the semantics are easy to get subtly wrong:

Tokens, not characters.
    Prefix reuse happens at the token level. Two prompts sharing a character
    prefix may tokenise differently at the boundary and share fewer tokens than
    they appear to.

Residency is inherited upward.
    If a replica holds a 900-token prefix, it necessarily holds every shorter
    prefix of it — they are the same cache blocks. So residency is marked on the
    whole root-to-node path, and the longest match for a replica is the deepest
    node on the query path where it is resident.

Matches round down to a block boundary.
    Engines cache in fixed-size blocks (vLLM defaults to 16 tokens). A replica
    holding 100 tokens can only serve 96 of them as a hit; the remainder has to
    be re-prefilled. Reporting the unrounded number is the single easiest way to
    overstate how well a routing policy is doing.

Eviction is tail-first.
    A prefix block cannot be freed while a longer prefix that extends it is still
    resident, because the longer one is physically built on top of it. So only
    nodes with no resident descendant are evictable. This mirrors how a real
    engine's block allocator behaves and it is what makes LRU non-trivial here.

The tree is the router's *belief* about replica state, not ground truth. It
drifts whenever a replica evicts something the router didn't predict. Measuring
that drift is a core question of this project, not an implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Tokens = tuple[int, ...]

DEFAULT_BLOCK_SIZE = 16


@dataclass(slots=True)
class Node:
    """One edge-compressed segment of the prefix tree."""

    edge: Tokens                                  # tokens from parent to here
    depth: int                                    # tokens from root to here
    parent: "Node | None" = None
    children: dict[int, "Node"] = field(default_factory=dict)
    residents: dict[str, float] = field(default_factory=dict)  # replica -> last access

    def is_tail_for(self, replica: str) -> bool:
        """True if no descendant of this node is resident for `replica`."""
        return all(replica not in child.residents for child in self.children.values())


class PrefixTree:
    """Fleet-wide view of which replica caches which prefix."""

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        capacity_tokens: dict[str, int] | None = None,
    ) -> None:
        self.block_size = block_size
        self.root = Node(edge=(), depth=0)
        self.capacity_tokens = dict(capacity_tokens or {})
        self.tokens_used: dict[str, int] = {}
        # eviction counters, for measuring cache-state drift later
        self.evictions: dict[str, int] = {}

    # -- traversal ---------------------------------------------------------

    def _descend(self, tokens: Tokens) -> tuple[Node, int]:
        """Walk as far down the query path as the tree allows.

        Returns the deepest *fully matched* node and how many query tokens were
        matched in total (which may exceed that node's depth if the walk stopped
        partway along an outgoing edge).
        """
        node = self.root
        i = 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            edge = child.edge
            j = 0
            while j < len(edge) and i + j < len(tokens) and edge[j] == tokens[i + j]:
                j += 1
            if j < len(edge):
                return node, i + j          # stopped mid-edge: child not reached
            node, i = child, i + j
        return node, i

    def _split(self, child: Node, at: int) -> Node:
        """Split `child`'s edge at offset `at`, returning the new intermediate node."""
        parent = child.parent
        assert parent is not None
        mid = Node(
            edge=child.edge[:at],
            depth=parent.depth + at,
            parent=parent,
            residents=dict(child.residents),   # ancestors inherit residency
        )
        parent.children[mid.edge[0]] = mid
        child.edge = child.edge[at:]
        child.depth = mid.depth + len(child.edge)
        child.parent = mid
        mid.children[child.edge[0]] = child
        return mid

    # -- queries -----------------------------------------------------------

    def match(self, tokens: Tokens) -> dict[str, int]:
        """Longest block-aligned prefix of `tokens` each replica can serve."""
        best: dict[str, int] = {}
        node = self.root
        i = 0
        while True:
            for replica, _ in node.residents.items():
                usable = (node.depth // self.block_size) * self.block_size
                if usable > best.get(replica, 0):
                    best[replica] = usable
            if i >= len(tokens):
                break
            child = node.children.get(tokens[i])
            if child is None:
                break
            edge = child.edge
            j = 0
            while j < len(edge) and i + j < len(tokens) and edge[j] == tokens[i + j]:
                j += 1
            if j < len(edge):
                # Stopped partway along a compressed edge. That edge stands for a
                # chain of implicit nodes, so anyone resident at `child` also holds
                # every position along it -- credit them with the i+j tokens the
                # query actually shares. Missing this silently reports no hit for
                # any query shorter than a cached prefix.
                for replica in child.residents:
                    usable = ((i + j) // self.block_size) * self.block_size
                    if usable > best.get(replica, 0):
                        best[replica] = usable
                break
            node, i = child, i + j
        return {r: n for r, n in best.items() if n > 0}

    # -- mutation ----------------------------------------------------------

    def insert(self, tokens: Tokens, replica: str, now: float) -> None:
        """Record that `replica` now holds `tokens` as a cached prefix."""
        if not tokens:
            return
        node, matched = self._descend(tokens)

        if matched > node.depth:               # stopped mid-edge; split it
            child = node.children[tokens[node.depth]]
            node = self._split(child, matched - node.depth)

        if matched < len(tokens):              # extend with the novel suffix
            rest = tokens[matched:]
            leaf = Node(edge=rest, depth=node.depth + len(rest), parent=node)
            node.children[rest[0]] = leaf
            node = leaf

        # mark residency along the whole path: holding a prefix means holding
        # every shorter prefix of it
        cursor: Node | None = node
        while cursor is not None and cursor is not self.root:
            if replica not in cursor.residents:
                self.tokens_used[replica] = (
                    self.tokens_used.get(replica, 0) + len(cursor.edge)
                )
            cursor.residents[replica] = now
            cursor = cursor.parent

        self._enforce_capacity(replica)

    def touch(self, tokens: Tokens, replica: str, now: float) -> None:
        """Refresh LRU timestamps along a prefix the replica just served."""
        node = self.root
        i = 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None or len(child.edge) > len(tokens) - i:
                break
            if tokens[i : i + len(child.edge)] != child.edge:
                break
            node, i = child, i + len(child.edge)
            if replica in node.residents:
                node.residents[replica] = now

    # -- eviction ----------------------------------------------------------

    def _enforce_capacity(self, replica: str) -> None:
        cap = self.capacity_tokens.get(replica)
        if cap is None:
            return
        while self.tokens_used.get(replica, 0) > cap:
            if not self._evict_one(replica):
                break

    def _evict_one(self, replica: str) -> bool:
        """Drop the least-recently-used evictable node for `replica`."""
        victim: Node | None = None
        oldest = float("inf")
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is self.root or replica not in node.residents:
                continue
            if not node.is_tail_for(replica):
                continue                        # a longer prefix still needs it
            ts = node.residents[replica]
            if ts < oldest:
                victim, oldest = node, ts
        if victim is None:
            return False
        del victim.residents[replica]
        self.tokens_used[replica] = self.tokens_used.get(replica, 0) - len(victim.edge)
        self.evictions[replica] = self.evictions.get(replica, 0) + 1
        return True

    def remove_replica(self, replica: str) -> None:
        """Forget a replica entirely (it died, or drained)."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            node.residents.pop(replica, None)
        self.tokens_used.pop(replica, None)

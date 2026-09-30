"""Host-tier GDN checkpoints: the second residence for the SAME boundary.

Deliberately NOT built on ./driver.Session: that harness diffs every operation against a
reference model of the currency, and the bank adds no new currency -- it adds a place for one.
Modeling it would mean writing a second model, which tests my model instead of this code. So
these drive HybridRadixCache directly and pin the invariants the bank could actually break:

  * a host checkpoint is a resume point exactly like a device one (match walks to it);
  * the device copy wins, and displacing a bank copy returns the slot to the caller;
  * the 'a leaf always carries a snapshot' cascade treats a bank copy as a snapshot, so an
    internal node whose device snapshot is gone survives as a resume point -- and when the
    KV under a bank copy is finally evicted, the slot goes back;
  * ``evict_host`` is LRU over unlocked bank copies and frees no KV;
  * a device snapshot evicted while the bank has a slot for it -- a free one, or an older
    bank copy's -- moves there instead of being dropped.
"""
from __future__ import annotations

import torch

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

PAGE = 4


def ids(*labels: int, tokens: int | None = None) -> torch.Tensor:
    """Whole labelled pages (page keying, not token keying), optionally cut to ``tokens``."""
    out: list[int] = []
    for lab in labels:
        out.extend([lab] + [7] * (PAGE - 1))
    t = torch.tensor(out, dtype=torch.int32)
    return t if tokens is None else t[:tokens]


def kv(n_tokens: int) -> torch.Tensor:
    return torch.arange(1, n_tokens + 1, dtype=torch.int32) * PAGE


def cache() -> HybridRadixCache:
    return HybridRadixCache(torch.device("cpu"), PAGE)


def three_deep():
    """[0,4)|host 5 -> [4,8)|host 6 -> [8,12)|host 7 from ONE 12-token insert."""
    c = cache()
    full = ids(1, 2, 3)
    c.insert(full, kv(12), 11)                       # device snapshot at 12
    for n, host in zip((ids(1), ids(1, 2)), (5, 6)):
        assert c.attach_host(n, host)
    return c


def test_host_checkpoint_is_a_resume_point():
    c = three_deep()
    assert c.mamba_host_evictable == 2 and c.mamba_evictable == 1

    m = c.match_prefix(ids(1))                       # nothing deeper than the bank copy at 4
    assert m.cached_len == 4 and m.mamba_host == 5 and m.mamba_value is None
    assert m.kv_indices.numel() == 4, "the KV under a bank copy is what makes it resumable"

    m2 = c.match_prefix(ids(1, 2))
    assert m2.cached_len == 8 and m2.mamba_host == 6

    m3 = c.match_prefix(ids(1, 2, 3))                # the device copy wins at its own boundary
    assert m3.cached_len == 12 and m3.mamba_value == 11 and m3.mamba_host is None
    c.check_integrity()


def test_attach_refuses_a_boundary_the_walk_did_not_reach():
    """A checkpoint describes the state AT a length; hanging it on a shallower node would resume
    a request with the state of different tokens, so it must be refused, not rounded."""
    c = cache()
    c.insert(ids(1, 2), kv(8), 11)
    assert not c.attach_host(ids(1, 2, 3), 5)   # 12 tokens asked, only 8 are in the tree
    assert c.attach_host(ids(1), 5)             # a real boundary accepts it once
    assert not c.attach_host(ids(1), 6)         # one boundary, one checkpoint
    c.check_integrity()


def test_device_donation_displaces_the_bank_copy():
    c = three_deep()
    freed: list[int] = []
    _, exist = c.insert(ids(1), kv(4), 12, freed)    # device copy at the bank boundary 4
    assert freed == [5] and not exist
    assert c.mamba_host_evictable == 1               # 4 is device-owned now, 8 still banked
    m = c.match_prefix(ids(1))
    assert m.cached_len == 4 and m.mamba_value == 12 and m.mamba_host is None
    c.check_integrity()


def test_bank_eviction_is_lru_and_frees_no_kv():
    c = three_deep()
    c.match_prefix(ids(1))                           # touch the shallowest: it is no longer LRU
    dropped = c.evict_host(1)
    assert dropped == [6], "the untouched boundary goes first"
    assert c.mamba_host_evictable == 1
    assert c.full_evictable + c.full_protected == 12, "dropping a bank copy costs no KV"
    assert c.match_prefix(ids(1, 2)).cached_len == 4, "8 is no longer resumable, 4 still is"
    c.check_integrity()


def test_cascade_keeps_the_bank_resume_point_then_reclaims_its_slot():
    """evict_mamba takes the device snapshot off the leaf; the parent's bank copy is a snapshot
    too, so the cascade must NOT delete it -- it is the resume point for [0,4). Only when its own
    KV goes does its bank slot go back."""
    c = cache()
    c.insert(ids(1, 2), kv(8), 11)                   # one node [0,8), device at 8
    assert c.attach_host(ids(1), 5)                  # splits: [0,4)|host 5 -> [4,8)|device 11
    before = c.mamba_host_evictable

    c.evict_mamba(1)                                 # frees device 11; leaf dies, cascade stops
    assert c.mamba_host_evictable == before, "the bank resume point survives"
    assert c.mamba_evictable == 0
    assert c.match_prefix(ids(1)).cached_len == 4

    er = c.evict_full(4)                             # the surviving bank copy is a leaf now
    assert er.host_slots == [5], "KV gone -> the checkpoint describing it is worthless"
    assert c.mamba_host_evictable == 0 and len(c._host_nodes()) == 0
    c.check_integrity()


def test_locked_bank_slot_survives_bank_pressure():
    c = three_deep()
    node = c.match_prefix(ids(1)).node
    c.inc_lock(node)                                 # what a restoring request holds
    # A restoring request's source slot is not recyclable; the bank sheds the OTHER copy.
    assert c.evict_host(1) == [6] and c.mamba_host_protected == 1 and c.mamba_host_evictable == 0
    assert c.match_prefix(ids(1)).cached_len == 4, "the locked resume point is intact"
    c.dec_lock(node)
    assert c.evict_host(1) == [5] and len(c._host_nodes()) == 0
    c.check_integrity()


def test_an_evicted_device_snapshot_moves_into_an_offered_bank_slot():
    """A prompt that fits one prefill chunk freezes no intermediate checkpoint, so its end-of-turn
    device snapshot is its only resume point: evicting it must move it into the bank, not drop it."""
    c = cache()
    c.insert(ids(1, 2), kv(8), 11)                   # one node [0,8), device at 8
    er = c.evict_mamba(1, host_slots=[5])
    assert er.demoted == [(11, 5)] and er.mamba_slots == [11]
    assert er.kv_indices.numel() == 0, "the KV stays: the bank copy resumes from it"
    m = c.match_prefix(ids(1, 2))
    assert m.cached_len == 8 and m.mamba_value is None and m.mamba_host == 5
    assert c.mamba_evictable == 0 and c.mamba_host_evictable == 1
    c.check_integrity()


def test_a_full_bank_gives_a_demotion_only_an_older_checkpoints_slot():
    """With no free slot, the evicted snapshot takes the slot of the least recently used bank
    checkpoint if that one is older -- and is dropped as before if every bank copy is newer."""
    for bank_older in (True, False):
        c = cache()
        c.insert(ids(1), kv(4), 11)                  # branch A: device at 4
        c.insert(ids(2, 3), kv(8), 12)               # branch B: device at 8
        assert c.attach_host(ids(2), 6)              # B's [0,4) in the bank
        device_a = c.match_prefix(ids(1)).node
        bank_b = c.match_prefix(ids(2)).node
        device_b = c.match_prefix(ids(2, 3)).node
        device_a.timestamp, device_b.timestamp = 20, 30   # A is the device LRU
        bank_b.timestamp = 10 if bank_older else 25
        er = c.evict_mamba(1, host_slots=[])
        if bank_older:
            assert er.demoted == [(11, 6)], "the older bank copy gives way"
            assert c.match_prefix(ids(1)).mamba_host == 6
            assert c.match_prefix(ids(2)).cached_len == 0, "B's [0,4) is no longer resumable"
        else:
            assert er.demoted == [] and er.mamba_slots == [11]
            assert c.match_prefix(ids(1)).cached_len == 0, "A was the oldest: dropped as before"
            assert c.match_prefix(ids(2)).mamba_host == 6
        c.check_integrity()


def test_a_node_that_keeps_a_bank_copy_is_not_given_a_second():
    """A device snapshot lands beside the bank copy of its boundary while a restore holds that
    copy. Evicting the device snapshot leaves the bank copy as the resume point: taking an offered
    slot as well would leak the one the node holds."""
    c = cache()
    c.insert(ids(1, 2, 3), kv(12), 12)
    assert c.attach_host(ids(1, 2), 6)
    restoring = c.match_prefix(ids(1, 2)).node
    c.inc_lock(restoring)
    c.insert(ids(1, 2), kv(8), 11, freed_host=[])    # the locked bank copy stays put
    c.dec_lock(restoring)
    assert restoring.mamba_value == 11 and restoring.mamba_host == 6
    leaf = c.match_prefix(ids(1, 2, 3)).node
    restoring.timestamp, leaf.timestamp = 10, 20
    er = c.evict_mamba(1, host_slots=[5])
    assert er.mamba_slots == [11] and er.demoted == []
    assert c.match_prefix(ids(1, 2)).mamba_host == 6
    c.check_integrity()


def test_without_a_bank_eviction_drops_the_snapshot_as_before():
    c = cache()
    c.insert(ids(1, 2), kv(8), 11)
    er = c.evict_mamba(1)
    assert er.demoted == [] and er.mamba_slots == [11] and er.kv_indices.numel() == 8
    assert c.match_prefix(ids(1, 2)).cached_len == 0
    c.check_integrity()


def test_one_chunk_turns_stay_resumable_past_the_device_slots():
    """Eight turns, each one prefill chunk long, while the card keeps four snapshots: with the
    bank a replay departing after any turn resumes at that turn's boundary; without it, only
    after the four turns whose snapshot the card kept. Every walk gives a chain's ancestors one
    timestamp, so which four is a heap tie; the count is what the bank decides."""
    turns = range(1, 9)
    for bank, resumable in ((True, 8), (False, 4)):
        c = cache()
        free_bank = list(range(1, 9))
        device_slots = 4
        for turn in turns:
            c.insert(ids(*range(1, turn + 1)), kv(4 * turn), 100 + turn)
            while c.mamba_evictable > device_slots:
                offered = free_bank[:1] if bank else None
                er = c.evict_mamba(1, host_slots=offered)
                used = {host for _, host in er.demoted}
                free_bank = [slot for slot in free_bank if slot not in used]
        resumed = [c.match_prefix(ids(*range(1, turn + 1), 99)).cached_len for turn in turns]
        assert sum(end == 4 * turn for turn, end in zip(turns, resumed)) == resumable, (bank, resumed)
        c.check_integrity()

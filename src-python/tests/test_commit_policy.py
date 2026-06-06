"""LocalAgreement-2: a word is committed only once two consecutive Whisper
hypotheses agree on it (beyond what's already committed)."""
from stt_providers.ondevice.commit_policy import LocalAgreement

def _w(*words):
    # build (word, start_ms, end_ms) tuples; timings are placeholders here
    return [(w, i * 100, i * 100 + 100) for i, w in enumerate(words)]

def test_nothing_commits_on_first_hypothesis():
    la = LocalAgreement()
    committed = la.add(_w("the", "cat"))
    assert [c[0] for c in committed] == []
    assert [p[0] for p in la.partial()] == ["the", "cat"]

def test_commits_agreed_prefix_of_two_hypotheses():
    la = LocalAgreement()
    la.add(_w("the", "cat"))
    committed = la.add(_w("the", "cat", "sat"))
    assert [c[0] for c in committed] == ["the", "cat"]
    assert [c[0] for c in la.committed] == ["the", "cat"]
    assert [p[0] for p in la.partial()] == ["sat"]

def test_disagreement_stops_commit_at_divergence():
    la = LocalAgreement()
    la.add(_w("i", "scream"))
    committed = la.add(_w("ice", "cream"))   # diverges at word 0
    assert [c[0] for c in committed] == []
    assert [c[0] for c in la.committed] == []

def test_incremental_commit_advances_monotonically():
    la = LocalAgreement()
    la.add(_w("a", "b"))
    la.add(_w("a", "b", "c"))          # commits a, b
    committed = la.add(_w("a", "b", "c", "d"))  # commits c
    assert [c[0] for c in committed] == ["c"]
    assert [c[0] for c in la.committed] == ["a", "b", "c"]

def test_reset_clears_state():
    la = LocalAgreement()
    la.add(_w("a", "b")); la.add(_w("a", "b"))
    la.reset()
    assert la.committed == []
    assert la.partial() == []

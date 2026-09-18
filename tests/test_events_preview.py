"""Tests for POST /events/preview: read-only what-if projection."""

import pytest
from fastapi.testclient import TestClient

import app.db as db_module
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated SQLite file."""
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "preview-test.db"))
    db_module.init_db()
    return TestClient(app)


def _ev(event_id, participant_id="alice", points=1, correction_of=None):
    payload = {
        "event_id": event_id,
        "participant_id": participant_id,
        "points": points,
    }
    if correction_of is not None:
        payload["correction_of"] = correction_of
    return payload


def _post(client, event_id, participant_id="alice", points=1, correction_of=None):
    r = client.post(
        "/events", json=_ev(event_id, participant_id, points, correction_of)
    )
    assert r.status_code == 201, r.text
    return r.json()


def _preview(client, base_revision, events):
    r = client.post(
        "/events/preview", json={"base_revision": base_revision, "events": events}
    )
    assert r.status_code == 200, r.text
    return r.json()


def _totals(body):
    return [
        (e["rank"], e["participant_id"], e["total_points"])
        for e in body["projected_leaderboard"]["entries"]
    ]


def test_empty_baseline_projects_drafts_only(client):
    body = _preview(client, 0, [_ev("d1", "alice", 10), _ev("d2", "bob", 5)])
    assert body["base_revision"] == 0
    assert body["projected_leaderboard"]["revision"] == 2
    assert _totals(body) == [(1, "alice", 10), (2, "bob", 5)]
    assert [
        c["participant_id"] for c in body["changes"]
    ] == ["alice", "bob"]
    assert body["changes"][0] == {
        "participant_id": "alice",
        "before": None,
        "after": {"rank": 1, "total_points": 10},
        "points_delta": 10,
    }


def test_drafts_apply_on_top_of_baseline(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 5)

    body = _preview(client, 2, [_ev("d1", "alice", 4)])
    assert body["projected_leaderboard"]["revision"] == 3
    assert _totals(body) == [(1, "alice", 14), (2, "bob", 5)]
    # alice gains 4 but keeps rank 1; bob is unchanged and must be omitted.
    assert [c["participant_id"] for c in body["changes"]] == ["alice"]
    assert body["changes"][0] == {
        "participant_id": "alice",
        "before": {"rank": 1, "total_points": 10},
        "after": {"rank": 1, "total_points": 14},
        "points_delta": 4,
    }


def test_only_events_visible_at_base_revision_are_the_baseline(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 5)
    _post(client, "e3", "carol", 20)

    # Pinned at revision 1: e2/e3 must not contribute.
    body = _preview(client, 1, [_ev("d1", "zoe", 7)])
    assert _totals(body) == [
        (1, "alice", 10),
        (2, "zoe", 7),
    ]


def test_projected_leaderboard_matches_real_leaderboard_after_equivalent_writes(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 5)

    drafts = [
        _ev("d1", "carol", 12),
        _ev("d2", "bob", 8),
    ]
    body = _preview(client, 2, drafts)

    for d in drafts:
        r = client.post("/events", json=d)
        assert r.status_code == 201

    real = client.get("/leaderboard").json()
    assert body["projected_leaderboard"]["revision"] == real["revision"] == 4
    projected = body["projected_leaderboard"]
    assert [e["participant_id"] for e in projected["entries"]] == [
        e["participant_id"] for e in real["entries"]
    ]
    assert [e["total_points"] for e in projected["entries"]] == [
        e["total_points"] for e in real["entries"]
    ]
    assert [e["rank"] for e in projected["entries"]] == [
        e["rank"] for e in real["entries"]
    ]


def test_draft_correction_of_baseline_event_reattributes_points(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 5)

    body = _preview(client, 2, [_ev("d1", "zoe", 12, correction_of="e1")])
    assert _totals(body) == [(1, "zoe", 12), (2, "bob", 5)]
    changes = {c["participant_id"]: c for c in body["changes"]}
    # alice leaves, zoe enters, bob is untouched.
    assert changes["alice"] == {
        "participant_id": "alice",
        "before": {"rank": 1, "total_points": 10},
        "after": None,
        "points_delta": -10,
    }
    assert changes["zoe"] == {
        "participant_id": "zoe",
        "before": None,
        "after": {"rank": 1, "total_points": 12},
        "points_delta": 12,
    }
    assert "bob" not in changes


def test_draft_correction_chain_within_batch(client):
    _post(client, "e1", "alice", 10)
    body = _preview(
        client,
        1,
        [
            _ev("d1", "alice", 12, correction_of="e1"),
            _ev("d2", "alice", 15, correction_of="d1"),
        ],
    )
    assert body["projected_leaderboard"]["revision"] == 3
    entries = body["projected_leaderboard"]["entries"]
    assert len(entries) == 1
    assert entries[0]["total_points"] == 15
    # Only the chain tip d2 is effective, with its virtual revision 3.
    assert entries[0]["effective_events"] == [
        {"event_id": "d2", "points": 15, "revision": 3, "correction_of": "d1"}
    ]


def test_draft_forks_an_already_corrected_baseline_chain(client):
    # Baseline: e1 (alice,10) is superseded by e2 (carol,7). A draft that
    # corrects e1 again creates a fork with two effective leaves.
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 4)
    _post(client, "e3", "carol", 7, correction_of="e1")

    body = _preview(client, 3, [_ev("d1", "dave", 20, correction_of="e1")])
    entries = {e["participant_id"]: e["total_points"]
               for e in body["projected_leaderboard"]["entries"]}
    assert entries == {"dave": 20, "carol": 7, "bob": 4}
    # Both fork leaves are independently effective.
    dave = next(e for e in body["projected_leaderboard"]["entries"]
                if e["participant_id"] == "dave")
    assert [x["event_id"] for x in dave["effective_events"]] == ["d1"]


def test_fork_then_chain_correction_on_a_draft_branch(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "carol", 7, correction_of="e1")

    body = _preview(
        client,
        2,
        [
            _ev("d1", "dave", 20, correction_of="e1"),
            _ev("d2", "dave", 25, correction_of="d1"),
        ],
    )
    entries = {e["participant_id"]: e["total_points"]
               for e in body["projected_leaderboard"]["entries"]}
    # e1 forks into e2 (carol 7) and d1->d2 (dave 25); d1 is superseded.
    assert entries == {"dave": 25, "carol": 7}


def test_tied_ranks_are_sequential_with_participant_id_tiebreak(client):
    _post(client, "e1", "alice", 7)
    body = _preview(
        client,
        1,
        [
            _ev("d1", "aaron", 7),
            _ev("d2", "bob", 5),
        ],
    )
    assert _totals(body) == [
        (1, "aaron", 7),
        (2, "alice", 7),
        (3, "bob", 5),
    ]


def test_changes_include_rank_only_moves_with_zero_delta(client):
    _post(client, "e1", "carol", 7)
    _post(client, "e2", "bob", 5)
    # A draft slots in between: carol keeps rank 1, bob drops 2 -> 3 with
    # unchanged points, zoe enters.
    body = _preview(client, 2, [_ev("d1", "zoe", 6)])
    changes = {c["participant_id"]: c for c in body["changes"]}
    assert changes["bob"]["before"] == {"rank": 2, "total_points": 5}
    assert changes["bob"]["after"] == {"rank": 3, "total_points": 5}
    assert changes["bob"]["points_delta"] == 0
    assert "carol" not in changes  # rank 1 before and after: omitted


def test_changes_sorted_by_participant_id(client):
    body = _preview(
        client,
        0,
        [_ev("d1", "zoe", 1), _ev("d2", "alice", 2), _ev("d3", "bob", 3)],
    )
    assert [c["participant_id"] for c in body["changes"]] == [
        "alice", "bob", "zoe"
    ]


def test_effective_events_carry_virtual_revisions_after_base(client):
    _post(client, "e1", "alice", 10)
    body = _preview(
        client,
        1,
        [_ev("d1", "alice", 5), _ev("d2", "bob", 8)],
    )
    alice = next(
        e for e in body["projected_leaderboard"]["entries"]
        if e["participant_id"] == "alice"
    )
    # Baseline event keeps its real revision 1; drafts get base+1, base+2...
    assert [x["revision"] for x in alice["effective_events"]] == [1, 2]
    bob = next(
        e for e in body["projected_leaderboard"]["entries"]
        if e["participant_id"] == "bob"
    )
    assert bob["effective_events"][0]["revision"] == 3


def test_event_id_clash_with_any_ledger_event_returns_409(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 5)
    # Clash with an event beyond the baseline snapshot is still a 409.
    r = client.post(
        "/events/preview",
        json={"base_revision": 1, "events": [_ev("e2", "zoe", 1)]},
    )
    assert r.status_code == 409
    # Same payload as the stored event: still a clash, no idempotent replay.
    r = client.post(
        "/events/preview",
        json={"base_revision": 2, "events": [_ev("e1", "alice", 10)]},
    )
    assert r.status_code == 409


def test_conflict_409_takes_precedence_over_bad_reference(client):
    _post(client, "e1", "alice", 10)
    r = client.post(
        "/events/preview",
        json={
            "base_revision": 1,
            "events": [_ev("e1", "alice", 99, correction_of="ghost")],
        },
    )
    assert r.status_code == 409


def test_duplicate_event_id_within_draft_batch_returns_422(client):
    r = client.post(
        "/events/preview",
        json={
            "base_revision": 0,
            "events": [_ev("dup", "alice", 1), _ev("x", "bob", 2),
                       _ev("dup", "carol", 3)],
        },
    )
    assert r.status_code == 422


def test_self_reference_returns_422(client):
    r = client.post(
        "/events/preview",
        json={"base_revision": 0, "events": [
            _ev("self", "alice", 1, correction_of="self")
        ]},
    )
    assert r.status_code == 422


def test_reference_to_unknown_event_returns_422(client):
    _post(client, "e1", "alice", 10)
    r = client.post(
        "/events/preview",
        json={"base_revision": 1, "events": [
            _ev("d1", "alice", 1, correction_of="ghost")
        ]},
    )
    assert r.status_code == 422


def test_reference_to_later_draft_returns_422(client):
    r = client.post(
        "/events/preview",
        json={
            "base_revision": 0,
            "events": [
                _ev("early", "alice", 1, correction_of="later"),
                _ev("later", "alice", 2),
            ],
        },
    )
    assert r.status_code == 422


def test_reference_to_event_beyond_baseline_returns_422(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 5)
    r = client.post(
        "/events/preview",
        json={"base_revision": 1, "events": [
            _ev("d1", "alice", 1, correction_of="e2")
        ]},
    )
    assert r.status_code == 422


def test_reference_to_earlier_draft_is_allowed(client):
    r = client.post(
        "/events/preview",
        json={
            "base_revision": 0,
            "events": [
                _ev("a", "alice", 10),
                _ev("b", "bob", 12, correction_of="a"),
            ],
        },
    )
    assert r.status_code == 200
    assert _totals(r.json()) == [(1, "bob", 12)]


def test_reference_to_baseline_event_is_allowed(client):
    _post(client, "e1", "alice", 10)
    r = client.post(
        "/events/preview",
        json={"base_revision": 1, "events": [
            _ev("d1", "bob", 12, correction_of="e1")
        ]},
    )
    assert r.status_code == 200
    assert _totals(r.json()) == [(1, "bob", 12)]


def test_base_revision_beyond_max_returns_404(client):
    _post(client, "e1", "alice", 10)
    r = client.post(
        "/events/preview",
        json={"base_revision": 2, "events": [_ev("d1", "alice", 1)]},
    )
    assert r.status_code == 404


def test_base_revision_0_on_empty_ledger_is_valid(client):
    r = client.post(
        "/events/preview",
        json={"base_revision": 0, "events": [_ev("d1", "alice", 1)]},
    )
    assert r.status_code == 200


def test_negative_base_revision_returns_422(client):
    r = client.post(
        "/events/preview",
        json={"base_revision": -1, "events": [_ev("d1", "alice", 1)]},
    )
    assert r.status_code == 422


def test_missing_base_revision_returns_422(client):
    r = client.post("/events/preview", json={"events": [_ev("d1", "alice", 1)]})
    assert r.status_code == 422


def test_non_integer_base_revision_returns_422(client):
    r = client.post(
        "/events/preview",
        json={"base_revision": "abc", "events": [_ev("d1", "alice", 1)]},
    )
    assert r.status_code == 422


def test_batch_size_limits(client):
    assert client.post(
        "/events/preview", json={"base_revision": 0, "events": []}
    ).status_code == 422
    too_many = [_ev(f"x{i}", "p", 1) for i in range(101)]
    assert client.post(
        "/events/preview", json={"base_revision": 0, "events": too_many}
    ).status_code == 422
    exactly_100 = [_ev(f"h{i}", "p", i) for i in range(100)]
    r = client.post(
        "/events/preview", json={"base_revision": 0, "events": exactly_100}
    )
    assert r.status_code == 200
    assert r.json()["projected_leaderboard"]["revision"] == 100


def test_invalid_event_fields_return_422(client):
    # empty event_id
    r = client.post(
        "/events/preview",
        json={"base_revision": 0, "events": [
            {"event_id": "", "participant_id": "alice", "points": 1}
        ]},
    )
    assert r.status_code == 422
    # missing participant_id
    r = client.post(
        "/events/preview",
        json={"base_revision": 0, "events": [
            {"event_id": "d1", "points": 1}
        ]},
    )
    assert r.status_code == 422


def test_preview_is_read_only_and_consumes_no_revision(client):
    _post(client, "e1", "alice", 10)
    before_events = client.get("/events").json()
    before_revision = client.get("/leaderboard").json()["revision"]

    for _ in range(3):
        body = _preview(
            client,
            1,
            [
                _ev("d1", "bob", 20),
                _ev("d2", "carol", 30, correction_of="e1"),
            ],
        )
        # Repeated identical previews are deterministic.
        assert body["projected_leaderboard"]["revision"] == 3

    assert client.get("/events").json() == before_events
    assert client.get("/leaderboard").json()["revision"] == before_revision
    assert client.get("/events/d1").status_code == 404
    assert client.get("/events/d2").status_code == 404


def test_failed_preview_does_not_write(client):
    _post(client, "e1", "alice", 10)
    for payload in (
        {"base_revision": 0, "events": [_ev("e1", "alice", 99)]},  # 409
        {"base_revision": 99, "events": [_ev("d1", "alice", 1)]},  # 404
        {"base_revision": 1, "events": [
            _ev("d1", "alice", 1, correction_of="ghost")]},        # 422
    ):
        assert client.post("/events/preview", json=payload).status_code in (
            404, 409, 422
        )
    assert [e["event_id"] for e in client.get("/events").json()] == ["e1"]
    assert client.get("/leaderboard").json()["revision"] == 1

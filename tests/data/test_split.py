from lexi_research.data.split import split_rows, strict_test_rows


def test_grouped_split_is_deterministic_and_has_no_target_leakage() -> None:
    rows = [
        {"req_uid": str(i), "target_norm": f"word-{i // 2}", "text": f"sentence {i}"}
        for i in range(100)
    ]
    first = split_rows(rows, seed=7, version="v1")
    second = split_rows(list(reversed(rows)), seed=7, version="v1")
    assert {(r["req_uid"], r["split"]) for r in first.rows} == {(r["req_uid"], r["split"]) for r in second.rows}
    groups = {}
    for row in first.rows:
        groups.setdefault(row["target_norm"], set()).add(row["split"])
    assert all(len(value) == 1 for value in groups.values())


def test_strict_test_excludes_text_seen_before() -> None:
    rows = [
        {"text": "same", "split": "train"}, {"text": "same", "split": "test"}, {"text": "new", "split": "test"}
    ]
    assert [row["text"] for row in strict_test_rows(rows)] == ["new"]

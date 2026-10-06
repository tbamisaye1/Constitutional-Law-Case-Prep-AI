from app.db.repository import _library_records_from_backup_payload


def test_v2_named_arguments_key_is_restorable():
    payload = {
        "version": 2,
        "library_records": [],
        "arguments": [
            {
                "kind": "arguments",
                "id": "main",
                "data": {"draftsBySide": {"petitioner": []}},
            }
        ],
        "notebook": [],
    }
    rows = _library_records_from_backup_payload(payload)
    assert any(r["kind"] == "arguments" and r["id"] == "main" for r in rows)


def test_incomplete_dict_snapshot_still_restores_arguments():
    board = {"draftsBySide": {"petitioner": [{"id": "alt-q2-ladder", "sections": []}]}}
    payload = {"library_records": {"arguments": board}}
    rows = _library_records_from_backup_payload(payload)
    assert len(rows) == 1
    assert rows[0]["kind"] == "arguments"
    assert rows[0]["data"]["draftsBySide"]["petitioner"][0]["id"] == "alt-q2-ladder"


def test_v1_array_library_records_still_works():
    payload = {
        "library_records": [
            {"kind": "notebook", "id": "main", "data": {"tree": []}},
            {"kind": "arguments", "id": "main", "data": {"draftsBySide": {}}},
        ]
    }
    rows = _library_records_from_backup_payload(payload)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"notebook", "arguments"}

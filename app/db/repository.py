"""
Reads and writes for workspace-scoped prep data.

The wire format deliberately matches the frontend store (camelCase keys,
`updatedAt` as epoch milliseconds, `deleted` as a boolean) so the client can
push a slice of its own state without translating every field. The camelCase to
column mapping lives in ENTITIES below, in one place, instead of being spread
across six near-identical upsert functions.

Conflict rule: last write wins on `updatedAt`. That is the right trade-off for
one person moving between a laptop and an iPad, and it is what the local-first
client expects. It is the wrong trade-off for two people editing the same note
at once, which is why nothing here pretends to merge text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from psycopg.types.json import Jsonb

from app.db.notebook_merge import merge_notebook_snapshots

# A single push is capped so one bad client cannot send an unbounded statement.
# The whole seeded library is a few hundred rows, so this is generous.
MAX_ROWS_PER_ENTITY = 2_000

# Critical prep docs: auto-backup when any of these kinds change.
_BACKUP_TRIGGER_KINDS = frozenset(
    {
        "notebook",
        "arguments",
        "guide_edits",
        "facts",
        "openings",
        "opinions",
        "case_facts",
        "cites",
        "timeline",
    }
)

# Text fields on opinion / fact cards. A thinner push must not erase longer prose.
_RICH_TEXT_KINDS = frozenset({"opinions", "case_facts"})
_RICH_TEXT_FIELDS = ("bodyHtml", "notes", "summary", "text", "holding", "rule")
# Arguments / notebook change often before competition; snapshot every 15 minutes.
_AUTO_BACKUP_MIN_INTERVAL_SECONDS = 15 * 60
_AUTO_BACKUP_KEEP = 400  # ~4 days of 15-min snapshots + headroom
_MANUAL_BACKUP_KEEP = 100

# Explicit top-level keys in every downloadable backup (also still in library_records).
_BACKUP_DOC_KINDS = (
    "arguments",
    "notebook",
    "guide_edits",
    "facts",
    "openings",
    "opinions",
    "case_facts",
    "cites",
    "timeline",
    "note_tabs",
)

# Category 3 ladder seed fingerprints (generated from the frontend seed file).
# A sync push that re-uploads seed text must not erase manual argument notes.
_CATEGORY3_SEED_PATH = Path(__file__).with_name("category3_ladder_seed.json")
try:
    _CATEGORY3_SEED = json.loads(_CATEGORY3_SEED_PATH.read_text(encoding="utf-8"))
except Exception:  # pragma: no cover - missing file should not break sync
    _CATEGORY3_SEED = {"sections": {}}


@dataclass(frozen=True)
class EntitySpec:
    """
    How one client collection maps onto one table.

    Attributes:
        name: Key used for this collection in the sync payload.
        table: Postgres table name. Never interpolated from user input.
        key_fields: Wire-name to column pairs that identify a row inside a
            workspace. Together with workspace_id they form the primary key.
        value_fields: Wire-name to column pairs for everything else.
        json_fields: Wire names whose values are stored as JSONB and must be
            wrapped before psycopg will send them.
    """

    name: str
    table: str
    key_fields: tuple[tuple[str, str], ...]
    value_fields: tuple[tuple[str, str], ...]
    json_fields: frozenset[str] = frozenset()

    @property
    def all_fields(self) -> tuple[tuple[str, str], ...]:
        return self.key_fields + self.value_fields


ENTITIES: tuple[EntitySpec, ...] = (
    EntitySpec(
        name="matters",
        table="matters",
        key_fields=(("id", "id"),),
        value_fields=(
            ("title", "title"),
            ("season", "season"),
            ("issues", "issues"),
        ),
        json_fields=frozenset({"issues"}),
    ),
    EntitySpec(
        name="cases",
        table="cases",
        key_fields=(("id", "id"),),
        value_fields=(
            ("name", "name"),
            ("cite", "cite"),
            ("year", "year"),
            ("issue", "issue"),
            ("tag", "tag"),
            ("usefulness", "usefulness"),
            ("headlineNote", "headline_note"),
            ("holding", "holding"),
            ("rule", "rule"),
            ("usePetitioner", "use_petitioner"),
            ("useRespondent", "use_respondent"),
            ("suggestedFile", "suggested_file"),
        ),
    ),
    EntitySpec(
        name="documents",
        table="documents",
        key_fields=(("id", "id"),),
        # blobPathname is intentionally absent. Only the upload endpoint may set
        # it, so a client cannot point a document row at someone else's blob.
        value_fields=(
            ("caseId", "case_id"),
            ("name", "name"),
            ("size", "size_bytes"),
            ("contentType", "content_type"),
        ),
    ),
    EntitySpec(
        name="annotations",
        table="annotations",
        key_fields=(("id", "id"),),
        value_fields=(
            ("caseId", "case_id"),
            ("fileId", "document_id"),
            ("page", "page"),
            ("kind", "kind"),
            ("quote", "quote"),
            ("text", "body"),
            ("rects", "rects"),
            ("pinned", "pinned"),
            ("color", "color"),
            ("topics", "topics"),
        ),
        json_fields=frozenset({"rects", "topics"}),
    ),
    EntitySpec(
        name="notes",
        table="notes",
        key_fields=(("caseId", "case_id"), ("layerId", "layer_id")),
        value_fields=(("html", "html"),),
    ),
    EntitySpec(
        name="library_records",
        table="library_records",
        key_fields=(("kind", "kind"), ("id", "id")),
        value_fields=(("data", "data"),),
        json_fields=frozenset({"data"}),
    ),
)

ENTITY_BY_NAME = {spec.name: spec for spec in ENTITIES}

# Client delete tombstones often only carry the primary key (id + deleted).
# Postgres still needs every NOT NULL column on INSERT of a never-seen row, so
# fill the gaps before we write. Empty strings are fine: the row is tombstoned
# and the pull path already exposes deleted=True to other devices.
_PUSH_DEFAULTS: dict[str, dict[str, Any]] = {
    "matters": {"title": "", "season": "", "issues": []},
    "cases": {
        "name": "",
        "cite": "",
        "year": "",
        "usefulness": "background",
        "headlineNote": "",
        "holding": "",
        "rule": "",
        "usePetitioner": "",
        "useRespondent": "",
        "suggestedFile": "",
    },
    "documents": {
        "caseId": "",
        "name": "",
        "size": 0,
        "contentType": "application/pdf",
    },
    "annotations": {
        "caseId": "",
        "page": 1,
        "kind": "page",
        "quote": "",
        "text": "",
        "pinned": False,
        "color": "gold",
        "topics": [],
    },
    "notes": {"html": ""},
    "library_records": {"data": {}},
}


def _with_push_defaults(spec: EntitySpec, row: dict[str, Any]) -> dict[str, Any]:
    """
    Replace missing or null required wire fields so an id-only tombstone inserts.

    Live rows from a healthy client already send these fields. This exists for
    the delete path in collectChanges(), which historically only sent `{id,
    deleted: true, updatedAt}` for annotations and documents.
    """
    defaults = _PUSH_DEFAULTS.get(spec.name)
    if not defaults:
        return row

    filled = dict(row)
    for field_name, default in defaults.items():
        if filled.get(field_name) is None:
            filled[field_name] = default
    return filled


def to_epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def from_epoch_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def touch_workspace(cursor, workspace_id: str) -> None:
    """Create the workspace row on first contact, otherwise bump last_seen_at."""
    cursor.execute(
        """
        INSERT INTO workspaces (id) VALUES (%s)
        ON CONFLICT (id) DO UPDATE SET last_seen_at = now()
        """,
        (workspace_id,),
    )


def _row_to_wire(spec: EntitySpec, row: dict[str, Any]) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    for field_name, column in spec.all_fields:
        wire[field_name] = row[column]
    wire["updatedAt"] = to_epoch_ms(row["updated_at"])
    wire["deleted"] = row["deleted_at"] is not None
    # Clients need this to know they can re-fetch PDF bytes from Blob after a
    # refresh or on another device. blob_pathname itself stays server-only.
    if spec.name == "documents" and "blob_pathname" in row:
        wire["stored"] = row["blob_pathname"] is not None
    return wire


def pull_changes(cursor, workspace_id: str, since_ms: int) -> dict[str, list[dict]]:
    """
    Every row in this workspace modified after `since_ms`.

    Tombstones come back with deleted=True so a client that was offline learns
    about deletions instead of pushing the row back.

    Args:
        cursor: Open cursor on a dict-row connection.
        workspace_id: Validated workspace UUID string.
        since_ms: Cursor from the client's previous pull, epoch milliseconds.
            Pass 0 for a full download.

    Returns:
        Collection name to list of wire rows, ordered oldest change first.
    """
    since = from_epoch_ms(since_ms)
    changes: dict[str, list[dict]] = {}

    for spec in ENTITIES:
        columns = ", ".join(column for _, column in spec.all_fields)
        # documents.blob_pathname is not a sync field (clients must not aim a
        # row at someone else's blob), but pull still needs it to set `stored`.
        extra = ", blob_pathname" if spec.name == "documents" else ""
        cursor.execute(
            f"""
            SELECT {columns}{extra}, updated_at, deleted_at
            FROM {spec.table}
            WHERE workspace_id = %s AND updated_at > %s
            ORDER BY updated_at
            """,
            (workspace_id, since),
        )
        changes[spec.name] = [_row_to_wire(spec, row) for row in cursor.fetchall()]

    return changes


def _push_values(
    spec: EntitySpec,
    workspace_id: str,
    row: dict[str, Any],
    now: datetime,
) -> tuple[Any, ...]:
    values: list[Any] = [workspace_id]

    for field_name, _ in spec.all_fields:
        value = row.get(field_name)
        if field_name in spec.json_fields and value is not None:
            value = Jsonb(value)
        values.append(value)

    # Clamp to server time. A client whose clock runs fast would otherwise write
    # an updated_at in the future and win every later conflict, including
    # against edits the user makes afterwards on another device.
    client_updated_at = row.get("updatedAt")
    updated_at = now
    if isinstance(client_updated_at, (int, float)):
        updated_at = min(from_epoch_ms(int(client_updated_at)), now)

    values.append(updated_at)
    values.append(now if row.get("deleted") else None)
    return tuple(values)


# How many prior library_records payloads to keep per (workspace, kind, id).
# High enough for competition week (frequent sync) without Neon PITR.
_LIBRARY_REVISION_KEEP = 120

# Marks an `existing` argument the caller did not supply, as opposed to None,
# which means the caller looked and the row does not exist.
_NOT_LOADED: Any = object()


def _load_library_record(
    cursor,
    workspace_id: str,
    kind: Any,
    record_id: Any,
) -> dict[str, Any] | None:
    """
    The stored library_records row (data, updated_at, deleted_at), or None.

    Neon bills for every byte a query returns, and Arguments / notebook rows
    are the largest in the workspace. push_changes loads each row once with
    this and hands it to every guard, instead of each guard selecting the
    full board again. Tombstones are returned too; callers that only care
    about live rows should pass the result through _live_record.
    """
    if not isinstance(kind, str) or not isinstance(record_id, str):
        return None
    cursor.execute(
        """
        SELECT data, updated_at, deleted_at
        FROM library_records
        WHERE workspace_id = %s AND kind = %s AND id = %s
        """,
        (workspace_id, kind, record_id),
    )
    return cursor.fetchone()


def _live_record(existing: dict[str, Any] | None) -> dict[str, Any] | None:
    """The row when it exists and is not a tombstone, otherwise None."""
    if not existing or existing.get("deleted_at") is not None:
        return None
    return existing


def _resolve_live_record(
    cursor,
    workspace_id: str,
    kind: Any,
    record_id: Any,
    existing: Any,
) -> dict[str, Any] | None:
    """Use the caller's preloaded row when given, otherwise load it."""
    if existing is _NOT_LOADED:
        existing = _load_library_record(cursor, workspace_id, kind, record_id)
    return _live_record(existing)


def _library_record_payload_unchanged(
    cursor,
    workspace_id: str,
    prepared: dict[str, Any],
    *,
    existing: Any = _NOT_LOADED,
) -> bool:
    """
    True when the incoming library_records row would not change stored data.

    Used to skip no-op upserts so updated_at stays put. Tombstones and missing
    rows always return False so create / delete still run.
    """
    kind = prepared.get("kind")
    record_id = prepared.get("id")
    incoming = prepared.get("data")
    if not isinstance(kind, str) or not isinstance(record_id, str):
        return False
    if prepared.get("deleted"):
        return False
    try:
        live = _resolve_live_record(cursor, workspace_id, kind, record_id, existing)
    except Exception:
        return False
    if not live:
        return False
    return live.get("data") == incoming


def _archive_library_record_before_overwrite(
    cursor,
    workspace_id: str,
    row: dict[str, Any],
    now: datetime,
    *,
    source: str = "sync_push",
    existing: Any = _NOT_LOADED,
) -> int | None:
    """
    Copy the current library_records.data into history when a push would change it.

    Sync itself stays last-write-wins. This archive exists so a wiped arguments
    board (or notebook / guide edits) can be pulled back without Neon PITR.

    Args:
        source: Who triggered the overwrite (`sync_push`, `mcp`, `restore`, …).
        existing: The stored row from _load_library_record, when the caller
            already has it. Omit to load it here.

    Returns:
        The new revision id when a row was archived, otherwise None.
    """
    kind = row.get("kind")
    record_id = row.get("id")
    if not isinstance(kind, str) or not isinstance(record_id, str):
        return None

    try:
        existing = _resolve_live_record(cursor, workspace_id, kind, record_id, existing)
        if not existing:
            return None

        incoming = row.get("data")
        if incoming is None:
            return None
        # Skip archive when the payload is identical; avoids noise on heartbeat syncs.
        if existing["data"] == incoming:
            return None

        cursor.execute(
            """
            INSERT INTO library_record_revisions (
                workspace_id, kind, record_id, data, row_updated_at, revised_at, source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                workspace_id,
                kind,
                record_id,
                Jsonb(existing["data"]),
                existing["updated_at"],
                now,
                (source or "sync_push")[:40],
            ),
        )
        revision_row = cursor.fetchone()
        cursor.execute(
            """
            DELETE FROM library_record_revisions
            WHERE id IN (
                SELECT id FROM library_record_revisions
                WHERE workspace_id = %s AND kind = %s AND record_id = %s
                ORDER BY revised_at DESC
                OFFSET %s
            )
            """,
            (workspace_id, kind, record_id, _LIBRARY_REVISION_KEEP),
        )
        return int(revision_row["id"]) if revision_row else None
    except Exception:
        # Never let history bookkeeping block a sync. Migration may not be
        # applied yet on a preview deploy; the upsert below still proceeds.
        return None


def push_changes(
    cursor,
    workspace_id: str,
    payload: dict[str, Sequence[dict]],
    now: datetime,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """
    Upsert client rows, keeping whichever version has the newer updatedAt.

    Args:
        cursor: Open cursor on a dict-row connection.
        workspace_id: Validated workspace UUID string.
        payload: Collection name to rows in wire format. Unknown collection
            names are ignored so an older backend does not reject a newer
            client outright.
        now: Server time used to clamp client clocks and stamp tombstones.

    Returns:
        (written, rejected). written maps collection name to rows actually
        written. rejected lists Arguments (and similar) rows where protect
        refused or rewrote the client's payload so the UI can warn.
    """
    written: dict[str, int] = {}
    rejected: list[dict[str, Any]] = []

    for name, rows in payload.items():
        spec = ENTITY_BY_NAME.get(name)
        if spec is None or not rows:
            continue

        columns = ["workspace_id"] + [column for _, column in spec.all_fields]
        columns += ["updated_at", "deleted_at"]
        placeholders = ", ".join(["%s"] * len(columns))
        conflict_target = ", ".join(
            ["workspace_id"] + [column for _, column in spec.key_fields]
        )
        # Key columns are excluded: they are what we conflicted on, so writing
        # them back is a no-op that only makes the statement harder to read.
        updatable = [column for _, column in spec.value_fields]
        updatable += ["updated_at", "deleted_at"]
        assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in updatable)

        statement = f"""
            INSERT INTO {spec.table} ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_target}) DO UPDATE SET {assignments}
            WHERE EXCLUDED.updated_at >= {spec.table}.updated_at
        """

        count = 0
        touched_backup_kinds = False
        for row in rows[:MAX_ROWS_PER_ENTITY]:
            prepared = _with_push_defaults(spec, row)
            reject_reason: str | None = None
            reject_server_row: dict[str, Any] | None = None
            if name == "library_records":
                if prepared.get("kind") in _BACKUP_TRIGGER_KINDS:
                    touched_backup_kinds = True
                # The guards below only read the stored row and none of them
                # writes it, so one load stays valid until the upsert.
                try:
                    existing = _load_library_record(
                        cursor, workspace_id, prepared.get("kind"), prepared.get("id")
                    )
                except Exception:
                    existing = _NOT_LOADED
                prepared, reject_reason, reject_server_row = (
                    _protect_arguments_stale_base(
                        cursor, workspace_id, prepared, existing=existing
                    )
                )
                if reject_reason:
                    # Stale / missing base: keep the live row untouched. Writing
                    # would bump updated_at and thrash every other client.
                    entry: dict[str, Any] = {
                        "collection": name,
                        "kind": prepared.get("kind"),
                        "id": prepared.get("id"),
                        "reason": reject_reason,
                    }
                    if reject_server_row is not None:
                        entry["data"] = reject_server_row.get("data")
                        entry["serverUpdatedAt"] = reject_server_row.get(
                            "serverUpdatedAt"
                        )
                    rejected.append(entry)
                    continue
                prepared = _protect_notebook_from_smaller_push(
                    cursor, workspace_id, prepared, existing=existing
                )
                prepared, reject_reason = _protect_arguments_from_seed_or_thinner_push(
                    cursor, workspace_id, prepared, existing=existing
                )
                prepared = _protect_rich_library_text_from_thinner_push(
                    cursor, workspace_id, prepared, existing=existing
                )
                if reject_reason:
                    # Stamp server time so the pull echoes the kept board and
                    # the client does not stay on a discarded local edit.
                    prepared = {**prepared, "updatedAt": to_epoch_ms(now)}
                    rejected.append(
                        {
                            "collection": name,
                            "kind": prepared.get("kind"),
                            "id": prepared.get("id"),
                            "reason": reject_reason,
                        }
                    )
                elif _library_record_payload_unchanged(
                    cursor, workspace_id, prepared, existing=existing
                ):
                    # Identical content must not bump updated_at. Idle Arguments
                    # tabs were re-pushing the same board every few seconds and
                    # winning the version race against MCP / other devices.
                    continue
                _archive_library_record_before_overwrite(
                    cursor, workspace_id, prepared, now, existing=existing
                )
            cursor.execute(
                statement,
                _push_values(spec, workspace_id, prepared, now),
            )
            count += cursor.rowcount
        written[name] = count
        if name == "library_records" and touched_backup_kinds and count:
            maybe_auto_backup_workspace(cursor, workspace_id, now, reason="sync_push")

    return written, rejected


_SYNC_AUDIT_KEEP = 500


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _c3_s1_a_notes_from_board(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    try:
        for draft in (data.get("draftsBySide") or {}).get("petitioner") or []:
            if not isinstance(draft, dict):
                continue
            for section in draft.get("sections") or []:
                if not isinstance(section, dict):
                    continue
                for prong in section.get("prongs") or []:
                    if isinstance(prong, dict) and prong.get("id") == "c3-s1-a":
                        return str(prong.get("notes") or "")
    except Exception:
        return ""
    return ""


def build_sync_audit_payload(
    cursor,
    workspace_id: str,
    changes: dict[str, Sequence[dict]],
    written: dict[str, int],
    *,
    client_since_ms: int | None,
    event: str = "push_then_pull",
) -> dict[str, Any]:
    """
    Summarise one /sync request for forensic storage.

    Heartbeats (empty changes) are recorded too so a wall of 200 OK cannot hide
    a frozen Arguments board again.
    """
    inbound: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    arguments_snapshot: dict[str, Any] | None = None
    arguments_unchanged: bool | None = None

    for name, rows in (changes or {}).items():
        if not rows:
            continue
        inbound[name] = len(rows)
        for row in list(rows)[:MAX_ROWS_PER_ENTITY]:
            if not isinstance(row, dict):
                continue
            summary: dict[str, Any] = {
                "collection": name,
                "id": row.get("id"),
                "updatedAt": row.get("updatedAt"),
                "deleted": bool(row.get("deleted")),
            }
            if name == "library_records":
                kind = row.get("kind")
                data = row.get("data")
                summary["kind"] = kind
                summary["bytes"] = len(
                    json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                ) if data is not None else 0
                summary["sha256"] = _sha256_json(data) if data is not None else None
                if kind == "arguments" and isinstance(data, dict):
                    arguments_snapshot = data
                    notes = _c3_s1_a_notes_from_board(data)
                    summary["c3s1aNotesSha256"] = (
                        hashlib.sha256(notes.encode("utf-8")).hexdigest() if notes else None
                    )
                    summary["c3s1aNotesBytes"] = len(notes.encode("utf-8")) if notes else 0
                    draft0 = ""
                    try:
                        draft0 = str(
                            (data.get("draftsBySide") or {})
                            .get("petitioner", [{}])[0]
                            .get("notes")
                            or ""
                        )[:120]
                    except Exception:
                        draft0 = ""
                    summary["draftNotesPrefix"] = draft0
                    # Compare to live server row so heartbeats that re-send the
                    # same board are visible as arguments_unchanged=true. The
                    # comparison runs in Postgres so only a boolean comes back,
                    # not the whole board (Neon bills for returned bytes).
                    try:
                        cursor.execute(
                            """
                            SELECT jsonb_typeof(data) = 'object' AND data = %s AS same
                            FROM library_records
                            WHERE workspace_id = %s AND kind = 'arguments' AND id = %s
                              AND deleted_at IS NULL
                            """,
                            (Jsonb(data), workspace_id, row.get("id") or "main"),
                        )
                        existing = cursor.fetchone()
                        arguments_unchanged = bool(existing and existing.get("same"))
                    except Exception:
                        arguments_unchanged = None
            else:
                # Compact hash of the whole row without storing full annotation text.
                summary["sha256"] = _sha256_json(row)
            summaries.append(summary)

    empty_push = not any(inbound.values())
    note_bits = []
    if empty_push:
        note_bits.append("empty_inbound")
    if arguments_unchanged is True:
        note_bits.append("arguments_identical_to_server")
    if arguments_unchanged is False and arguments_snapshot is not None:
        note_bits.append("arguments_differs_from_server")
    if written.get("library_records"):
        note_bits.append(f"wrote_library_records={written.get('library_records')}")

    return {
        "event": event,
        "client_since_ms": client_since_ms,
        "inbound": inbound,
        "written": written,
        "row_summaries": summaries,
        "arguments_snapshot": arguments_snapshot,
        "empty_push": empty_push,
        "arguments_unchanged": arguments_unchanged,
        "note": ",".join(note_bits) if note_bits else None,
    }


def record_sync_audit(
    cursor,
    workspace_id: str,
    audit: dict[str, Any],
    now: datetime | None = None,
) -> None:
    """
    Persist one sync forensic row. Never raises into the request path.
    """
    stamp = now or datetime.now(timezone.utc)
    try:
        cursor.execute(
            """
            INSERT INTO sync_audit_log (
                workspace_id, created_at, event, client_since_ms,
                inbound, written, row_summaries, arguments_snapshot,
                empty_push, arguments_unchanged, note
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                workspace_id,
                stamp,
                str(audit.get("event") or "push_then_pull")[:80],
                audit.get("client_since_ms"),
                Jsonb(audit.get("inbound") or {}),
                Jsonb(audit.get("written") or {}),
                Jsonb(audit.get("row_summaries") or []),
                Jsonb(audit["arguments_snapshot"])
                if isinstance(audit.get("arguments_snapshot"), dict)
                else None,
                bool(audit.get("empty_push")),
                audit.get("arguments_unchanged"),
                (audit.get("note") or None),
            ),
        )
        cursor.execute(
            """
            DELETE FROM sync_audit_log
            WHERE id IN (
                SELECT id FROM sync_audit_log
                WHERE workspace_id = %s
                ORDER BY created_at DESC
                OFFSET %s
            )
            """,
            (workspace_id, _SYNC_AUDIT_KEEP),
        )
    except Exception:
        # Table may not exist until migrate runs; never block sync.
        return


def list_sync_audit(
    cursor,
    workspace_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Recent sync forensic rows for this workspace (newest first)."""
    limit = max(1, min(int(limit or 50), 200))
    cursor.execute(
        """
        SELECT id, created_at, event, client_since_ms, inbound, written,
               row_summaries, arguments_snapshot IS NOT NULL AS has_arguments_snapshot,
               empty_push, arguments_unchanged, note
        FROM sync_audit_log
        WHERE workspace_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (workspace_id, limit),
    )
    rows = []
    for row in cursor.fetchall():
        rows.append(
            {
                "id": row["id"],
                "createdAt": to_epoch_ms(row["created_at"]) if row["created_at"] else None,
                "event": row["event"],
                "clientSinceMs": row["client_since_ms"],
                "inbound": row["inbound"] or {},
                "written": row["written"] or {},
                "rowSummaries": row["row_summaries"] or [],
                "hasArgumentsSnapshot": bool(row["has_arguments_snapshot"]),
                "emptyPush": bool(row["empty_push"]),
                "argumentsUnchanged": row["arguments_unchanged"],
                "note": row["note"],
            }
        )
    return rows


def _protect_notebook_from_smaller_push(
    cursor,
    workspace_id: str,
    prepared: dict[str, Any],
    *,
    existing: Any = _NOT_LOADED,
) -> dict[str, Any]:
    """
    Merge an incoming notebook row with the server copy when the push would
    drop sections that still have pages. Sync stays last-write-wins for timing;
    content merge only fills gaps the wiped client no longer has.
    """
    if prepared.get("kind") != "notebook":
        return prepared
    incoming = prepared.get("data")
    if not isinstance(incoming, dict):
        return prepared

    try:
        existing = _resolve_live_record(
            cursor, workspace_id, "notebook", prepared.get("id"), existing
        )
        if not existing or not isinstance(existing.get("data"), dict):
            return prepared
        merged = merge_notebook_snapshots(incoming, existing["data"])
        if merged != incoming:
            return {**prepared, "data": merged}
    except Exception:
        return prepared
    return prepared


def _norm_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _seed_prong_notes(prong_id: str) -> str:
    for section in (_CATEGORY3_SEED.get("sections") or {}).values():
        prongs = section.get("prongs") or {}
        if prong_id in prongs:
            return str(prongs[prong_id].get("notes") or "")
    return ""


def _seed_prong_title(prong_id: str) -> str:
    for section in (_CATEGORY3_SEED.get("sections") or {}).values():
        prongs = section.get("prongs") or {}
        if prong_id in prongs:
            return str(prongs[prong_id].get("title") or "").strip()
    return ""


def _looks_like_seed_notes(prong_id: str, notes: str) -> bool:
    norm = _norm_html(notes)
    if not norm:
        return False
    # Misplaced / old seed block that kept showing up under 2.1 in the UI.
    if "Concede Hamdi on its own facts immediately" in norm and "The four sources they stack" in norm:
        return True
    # Bundled Category 3 opener. Treat any Walk-the-three-steps heading as seed
    # unless the text also carries the agent "claim in one sentence" fingerprint
    # (that board was user/restore content, not the starter outline).
    norm_lower = norm.lower()
    if "walk the three steps" in norm_lower and "claim in one sentence" not in norm_lower:
        return True
    seed = _seed_prong_notes(prong_id)
    if not seed:
        return False
    seed_norm = _norm_html(seed)
    if not seed_norm:
        return False
    if norm == seed_norm or norm.startswith(seed_norm[:80]):
        return True
    # Seed rewrite often keeps the first <h2>; treat that heading as a fingerprint.
    match = re.search(r"<h2[^>]*>(.*?)</h2>", seed, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return False
    heading = _norm_html(match.group(1))
    # "Walk the three steps" is also in older seed; do not treat the user's
    # "claim in one sentence" board as seed just because a later seed file
    # reused Jackson language.
    if heading.lower().startswith("walk the three steps") and "claim in one sentence" in norm.lower():
        return False
    return bool(heading) and len(heading) >= 12 and heading in norm


def _pick_argument_notes(server_notes: str, incoming_notes: str, prong_id: str) -> str:
    """
    Prefer the client's newest edit.

    Longer-wins used to silently undo trims and clears. Only block the bundled
    Category 3 seed and the agent Jackson restore fingerprint when those would
    overwrite real notes.
    """
    server = server_notes if isinstance(server_notes, str) else ""
    incoming = incoming_notes if isinstance(incoming_notes, str) else ""
    if server == incoming:
        return incoming
    # Empty incoming is a deliberate clear — allow it.
    if not incoming.strip():
        return incoming
    if not server.strip():
        return incoming
    # Agent "claim in one sentence" restore must not beat manual prong notes.
    if prong_id == "c3-s1-a":
        server_ai = _looks_like_ai_prong_notes(server)
        incoming_ai = _looks_like_ai_prong_notes(incoming)
        if incoming_ai and not server_ai:
            return server
        if server_ai and not incoming_ai:
            return incoming
    server_seed = _looks_like_seed_notes(prong_id, server)
    incoming_seed = _looks_like_seed_notes(prong_id, incoming)
    if incoming_seed and not server_seed:
        return server
    if server_seed and not incoming_seed:
        return incoming
    return incoming


def _looks_like_seed_draft_notes(notes: str) -> bool:
    """Whole-argument / section notes that match the bundled Category 3 seed."""
    norm = _norm_html(notes)
    if not norm:
        return False
    if "We ask this court to reverse for 3 reasons" in norm and "The ladder (say this in the roadmap)" in norm:
        return True
    if norm.startswith("Introduction The Constitution lets the President turn the armed forces"):
        return True
    draft_seed = str(_CATEGORY3_SEED.get("draftNotes") or "")
    if draft_seed:
        seed_norm = _norm_html(draft_seed)
        if seed_norm and (norm == seed_norm or norm.startswith(seed_norm[:100])):
            return True
    return False


def _pick_rich_notes(server_notes: str, incoming_notes: str) -> str:
    """
    Prefer the client's newest whole-argument / section notes.

    Seed outline text still loses to manual notes. Length is not a signal.
    """
    server = server_notes if isinstance(server_notes, str) else ""
    incoming = incoming_notes if isinstance(incoming_notes, str) else ""
    if server == incoming:
        return incoming
    if not incoming.strip():
        return incoming
    if not server.strip():
        return incoming
    server_seed = _looks_like_seed_draft_notes(server)
    incoming_seed = _looks_like_seed_draft_notes(incoming)
    if incoming_seed and not server_seed:
        return server
    if server_seed and not incoming_seed:
        return incoming
    return incoming


# Agent-written title/notes that must never beat the user's outline.
# "Jackson's method / claim in one sentence" was restore fodder, not Tobi's typing.
_AI_C3_S1_A_TITLES = {
    "a. Jackson’s method, not just his labels",
    "a. Jackson's method, not just his labels",
}


def _looks_like_ai_prong_notes(notes: str) -> bool:
    norm = _norm_html(notes)
    if not norm:
        return False
    if "The claim in one sentence" in norm and "Where Congress has legislated" in norm:
        return True
    return False


def _pick_argument_title(server_title: str, incoming_title: str, prong_id: str) -> str:
    server = server_title.strip() if isinstance(server_title, str) else ""
    incoming = incoming_title.strip() if isinstance(incoming_title, str) else ""
    if server == incoming:
        return incoming_title if isinstance(incoming_title, str) else incoming
    # Empty title is a deliberate clear.
    if not incoming:
        return incoming_title if isinstance(incoming_title, str) else incoming
    if not server:
        return incoming_title if isinstance(incoming_title, str) else incoming

    # c3-s1-a: never let agent Jackson-method title overwrite the user's title.
    if prong_id == "c3-s1-a":
        if incoming in _AI_C3_S1_A_TITLES and server not in _AI_C3_S1_A_TITLES:
            return server_title if isinstance(server_title, str) else server
        if server in _AI_C3_S1_A_TITLES and incoming not in _AI_C3_S1_A_TITLES:
            return incoming_title if isinstance(incoming_title, str) else incoming

    seed_title = _seed_prong_title(prong_id)
    if seed_title:
        if incoming == seed_title and server != seed_title:
            return server_title if isinstance(server_title, str) else server
        if server == seed_title and incoming != seed_title:
            return incoming_title if isinstance(incoming_title, str) else incoming
    return incoming_title if isinstance(incoming_title, str) else incoming


def _merge_arguments_boards(incoming: dict[str, Any], server: dict[str, Any]) -> dict[str, Any]:
    """
    Keep manual draft / section / prong notes when an incoming sync re-applies seed.
    Outline deletions on the incoming board still win (missing prong ids stay gone).
    Newest non-seed / non-agent edits win, including shorter text and clears.
    """
    incoming_sides = incoming.get("draftsBySide")
    server_sides = server.get("draftsBySide")
    if not isinstance(incoming_sides, dict) or not isinstance(server_sides, dict):
        return incoming

    merged_sides: dict[str, Any] = {}
    changed = False
    for side, incoming_drafts in incoming_sides.items():
        if not isinstance(incoming_drafts, list):
            merged_sides[side] = incoming_drafts
            continue
        server_drafts = server_sides.get(side) if isinstance(server_sides.get(side), list) else []
        server_by_id = {
            d.get("id"): d for d in server_drafts if isinstance(d, dict) and d.get("id")
        }
        next_drafts = []
        for draft in incoming_drafts:
            if not isinstance(draft, dict):
                next_drafts.append(draft)
                continue
            server_draft = server_by_id.get(draft.get("id"))
            if not isinstance(server_draft, dict):
                next_drafts.append(draft)
                continue
            draft_notes = _pick_rich_notes(
                str(server_draft.get("notes") or ""),
                str(draft.get("notes") or ""),
            )
            if draft_notes != draft.get("notes"):
                changed = True
            server_sections = {
                s.get("id"): s
                for s in (server_draft.get("sections") or [])
                if isinstance(s, dict) and s.get("id")
            }
            next_sections = []
            for section in draft.get("sections") or []:
                if not isinstance(section, dict):
                    next_sections.append(section)
                    continue
                server_section = server_sections.get(section.get("id"))
                if not isinstance(server_section, dict):
                    next_sections.append(section)
                    continue
                section_notes = _pick_rich_notes(
                    str(server_section.get("notes") or ""),
                    str(section.get("notes") or ""),
                )
                if section_notes != section.get("notes"):
                    changed = True
                server_prongs = {
                    p.get("id"): p
                    for p in (server_section.get("prongs") or [])
                    if isinstance(p, dict) and p.get("id")
                }
                next_prongs = []
                for prong in section.get("prongs") or []:
                    if not isinstance(prong, dict):
                        next_prongs.append(prong)
                        continue
                    server_prong = server_prongs.get(prong.get("id"))
                    if not isinstance(server_prong, dict):
                        next_prongs.append(prong)
                        continue
                    prong_id = str(prong.get("id"))
                    notes = _pick_argument_notes(
                        str(server_prong.get("notes") or ""),
                        str(prong.get("notes") or ""),
                        prong_id,
                    )
                    title = _pick_argument_title(
                        str(server_prong.get("title") or ""),
                        str(prong.get("title") or ""),
                        prong_id,
                    )
                    if notes != prong.get("notes") or title != prong.get("title"):
                        changed = True
                    next_prongs.append({**prong, "notes": notes, "title": title})
                next_sections.append({**section, "notes": section_notes, "prongs": next_prongs})
            next_drafts.append({**draft, "notes": draft_notes, "sections": next_sections})
        merged_sides[side] = next_drafts

    if not changed:
        return incoming
    return {**incoming, "draftsBySide": merged_sides}


def _c3_s1_a_title(board: dict[str, Any]) -> str:
    try:
        drafts = (board.get("draftsBySide") or {}).get("petitioner") or []
        for draft in drafts:
            if not isinstance(draft, dict):
                continue
            for section in draft.get("sections") or []:
                if not isinstance(section, dict) or section.get("id") != "c3-s1":
                    continue
                for prong in section.get("prongs") or []:
                    if isinstance(prong, dict) and prong.get("id") == "c3-s1-a":
                        return str(prong.get("title") or "").strip()
    except Exception:
        return ""
    return ""


def _protect_arguments_stale_base(
    cursor,
    workspace_id: str,
    prepared: dict[str, Any],
    *,
    existing: Any = _NOT_LOADED,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    """
    Reject Arguments pushes that did not start from the live server version.

    A dirty browser tab stamps Date.now() on every edit (or false dirty), so
    last-write-wins treats stale content as newer than an MCP / other-device
    write from a second earlier. Clients must send baseUpdatedAt equal to the
    server updated_at they last loaded; differing content without that match
    is refused and the live row is left alone.

    Returns (prepared, reason, server_row_or_none). server_row carries data +
    serverUpdatedAt so the client can drop its dirty board without a write.
    """
    if prepared.get("kind") != "arguments" or prepared.get("deleted"):
        return prepared, None, None
    incoming = prepared.get("data")
    if not isinstance(incoming, dict):
        return prepared, None, None
    try:
        existing = _resolve_live_record(
            cursor, workspace_id, "arguments", prepared.get("id"), existing
        )
        if not existing or not isinstance(existing.get("data"), dict):
            return prepared, None, None
        server = existing["data"]
        if incoming == server:
            return prepared, None, None
        server_ms = to_epoch_ms(existing["updated_at"])
        server_row = {"data": server, "serverUpdatedAt": server_ms}
        base = prepared.get("baseUpdatedAt")
        if base is None:
            return prepared, "arguments_rejected_missing_base", server_row
        try:
            base_ms = int(base)
        except (TypeError, ValueError):
            base_ms = -1
        if base_ms != int(server_ms):
            return prepared, "arguments_rejected_stale_base", server_row
    except Exception:
        return prepared, None, None
    return prepared, None, None


def _protect_arguments_from_seed_or_thinner_push(
    cursor,
    workspace_id: str,
    prepared: dict[str, Any],
    *,
    existing: Any = _NOT_LOADED,
) -> tuple[dict[str, Any], str | None]:
    """
    Block Category 3 seed / agent Jackson restores from erasing manual notes.

    Returns (prepared_row, rejection_reason). Rejection reason is set when the
    server refuses or rewrites any part of the incoming Arguments board so the
    client can warn instead of silently diverging.
    """
    if prepared.get("kind") != "arguments":
        return prepared, None
    incoming = prepared.get("data")
    if not isinstance(incoming, dict):
        return prepared, None
    try:
        existing = _resolve_live_record(
            cursor, workspace_id, "arguments", prepared.get("id"), existing
        )
        if not existing or not isinstance(existing.get("data"), dict):
            return prepared, None
        server = existing["data"]
        server_title = _c3_s1_a_title(server)
        incoming_title = _c3_s1_a_title(incoming)
        # Refuse whole-board overwrite when a tab pushes agent Jackson title
        # over the user's Youngstown / manual outline title.
        if (
            server_title not in _AI_C3_S1_A_TITLES
            and incoming_title in _AI_C3_S1_A_TITLES
        ):
            return {**prepared, "data": server}, "arguments_rejected_agent_jackson_title"

        def _c3_notes(board: dict[str, Any]) -> str:
            try:
                for draft in (board.get("draftsBySide") or {}).get("petitioner") or []:
                    if not isinstance(draft, dict):
                        continue
                    for section in draft.get("sections") or []:
                        if not isinstance(section, dict):
                            continue
                        for prong in section.get("prongs") or []:
                            if isinstance(prong, dict) and prong.get("id") == "c3-s1-a":
                                return str(prong.get("notes") or "")
            except Exception:
                return ""
            return ""

        server_notes = _c3_notes(server)
        incoming_notes = _c3_notes(incoming)
        # Whole-board refuse only when the push is the agent restore fingerprint
        # and the server does not already have that fingerprint. Field merge
        # handles mixed boards; length never decides.
        if _looks_like_ai_prong_notes(incoming_notes) and not _looks_like_ai_prong_notes(
            server_notes
        ):
            return {**prepared, "data": server}, "arguments_rejected_agent_claim_notes"

        merged = _merge_arguments_boards(incoming, server)
        if merged != incoming:
            return (
                {**prepared, "data": merged},
                "arguments_rejected_seed_content",
            )
    except Exception:
        return prepared, None
    return prepared, None


def _protect_rich_library_text_from_thinner_push(
    cursor,
    workspace_id: str,
    prepared: dict[str, Any],
    *,
    existing: Any = _NOT_LOADED,
) -> dict[str, Any]:
    """
    Keep longer opinion / case-fact prose when a newer but thinner client push
    would otherwise blank it (seed reload, partial localStorage, hard refresh).
    """
    kind = prepared.get("kind")
    if kind not in _RICH_TEXT_KINDS:
        return prepared
    incoming = prepared.get("data")
    if not isinstance(incoming, dict):
        return prepared

    try:
        existing = _resolve_live_record(
            cursor, workspace_id, kind, prepared.get("id"), existing
        )
        if not existing or not isinstance(existing.get("data"), dict):
            return prepared
        server = existing["data"]
        merged = dict(incoming)
        changed = False
        for field in _RICH_TEXT_FIELDS:
            incoming_text = merged.get(field)
            server_text = server.get(field)
            if not isinstance(server_text, str) or not server_text.strip():
                continue
            if not isinstance(incoming_text, str) or len(server_text) > len(incoming_text) + 40:
                merged[field] = server_text
                changed = True
        # Opinions UI reads bodyHtml || notes — keep them aligned when one side has prose.
        if kind == "opinions":
            body = merged.get("bodyHtml") if isinstance(merged.get("bodyHtml"), str) else ""
            notes = merged.get("notes") if isinstance(merged.get("notes"), str) else ""
            if len(body) > len(notes) + 20:
                merged["notes"] = body
                changed = True
            elif len(notes) > len(body) + 20:
                merged["bodyHtml"] = notes
                changed = True
        if changed:
            return {**prepared, "data": merged}
    except Exception:
        return prepared
    return prepared


def collect_workspace_backup_payload(cursor, workspace_id: str) -> dict[str, Any]:
    """
    Build a downloadable / restorable snapshot of everything that must survive
    a hard refresh or a bad sync (not PDF bytes — those live in Blob).

    Arguments, notebook, guide, facts, openings, and the rest of library_records
    are included twice: once in the full `library_records` array, and again as
    named top-level keys so a JSON download is obviously complete.
    """
    cursor.execute(
        """
        SELECT kind, id, data, updated_at
        FROM library_records
        WHERE workspace_id = %s AND deleted_at IS NULL
        ORDER BY kind, id
        """,
        (workspace_id,),
    )
    library_records = [
        {
            "kind": row["kind"],
            "id": row["id"],
            "data": row["data"],
            "updatedAt": to_epoch_ms(row["updated_at"]) if row["updated_at"] else None,
        }
        for row in cursor.fetchall()
    ]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in library_records:
        by_kind.setdefault(str(record["kind"]), []).append(record)

    cursor.execute(
        """
        SELECT id, case_id, document_id, page, kind, quote, body, rects, pinned,
               color, topics, updated_at
        FROM annotations
        WHERE workspace_id = %s AND deleted_at IS NULL
        ORDER BY updated_at
        """,
        (workspace_id,),
    )
    annotations = [
        {
            "id": row["id"],
            "caseId": row["case_id"],
            "fileId": row["document_id"],
            "page": row["page"],
            "kind": row["kind"],
            "quote": row["quote"],
            "text": row["body"],
            "rects": row["rects"],
            "pinned": row["pinned"],
            "color": row["color"],
            "topics": row["topics"],
            "updatedAt": to_epoch_ms(row["updated_at"]) if row["updated_at"] else None,
        }
        for row in cursor.fetchall()
    ]

    cursor.execute(
        """
        SELECT case_id, layer_id, html, updated_at
        FROM notes
        WHERE workspace_id = %s AND deleted_at IS NULL
        ORDER BY case_id, layer_id
        """,
        (workspace_id,),
    )
    notes = [
        {
            "caseId": row["case_id"],
            "layerId": row["layer_id"],
            "html": row["html"],
            "updatedAt": to_epoch_ms(row["updated_at"]) if row["updated_at"] else None,
        }
        for row in cursor.fetchall()
    ]

    cursor.execute(
        """
        SELECT id, name, cite, year, issue, tag, usefulness, holding, rule,
               use_petitioner, use_respondent, suggested_file, updated_at
        FROM cases
        WHERE workspace_id = %s AND deleted_at IS NULL
        ORDER BY id
        """,
        (workspace_id,),
    )
    cases = [
        {
            "id": row["id"],
            "name": row["name"],
            "cite": row["cite"],
            "year": row["year"],
            "issue": row["issue"],
            "tag": row["tag"],
            "usefulness": row["usefulness"],
            "holding": row["holding"],
            "rule": row["rule"],
            "usePetitioner": row["use_petitioner"],
            "useRespondent": row["use_respondent"],
            "suggestedFile": row["suggested_file"],
            "updatedAt": to_epoch_ms(row["updated_at"]) if row["updated_at"] else None,
        }
        for row in cursor.fetchall()
    ]

    cursor.execute(
        """
        SELECT id, case_id, name, size_bytes, content_type, blob_pathname,
               blob_url, updated_at
        FROM documents
        WHERE workspace_id = %s AND deleted_at IS NULL
        ORDER BY id
        """,
        (workspace_id,),
    )
    documents = [
        {
            "id": row["id"],
            "caseId": row["case_id"],
            "name": row["name"],
            "size": row["size_bytes"],
            "contentType": row["content_type"],
            "stored": row["blob_pathname"] is not None,
            "blobUrl": row["blob_url"],
            "updatedAt": to_epoch_ms(row["updated_at"]) if row["updated_at"] else None,
        }
        for row in cursor.fetchall()
    ]

    named_docs = {kind: by_kind.get(kind, []) for kind in _BACKUP_DOC_KINDS}
    includes = [
        kind for kind, rows in named_docs.items() if rows
    ] + [
        name
        for name, rows in (
            ("annotations", annotations),
            ("notes", notes),
            ("cases", cases),
            ("documents", documents),
        )
        if rows
    ]

    return {
        "version": 2,
        "workspaceId": workspace_id,
        "includes": includes,
        "library_records": library_records,
        **named_docs,
        "annotations": annotations,
        "notes": notes,
        "cases": cases,
        "documents": documents,
    }


def create_workspace_backup(
    cursor,
    workspace_id: str,
    *,
    label: str,
    source: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Insert a full workspace snapshot and prune old auto/manual rows."""
    stamp = now or datetime.now(timezone.utc)
    payload = collect_workspace_backup_payload(cursor, workspace_id)
    cursor.execute(
        """
        INSERT INTO workspace_backups (workspace_id, label, source, payload, created_at)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, label, source, created_at
        """,
        (workspace_id, label[:200], source[:80], Jsonb(payload), stamp),
    )
    row = cursor.fetchone()
    _prune_workspace_backups(cursor, workspace_id, source)
    includes = payload.get("includes") if isinstance(payload, dict) else []
    return {
        "id": row["id"],
        "label": row["label"],
        "source": row["source"],
        "createdAt": to_epoch_ms(row["created_at"]),
        "includes": includes or [],
        "hasArguments": bool(
            isinstance(payload, dict) and payload.get("arguments")
        ),
    }


def maybe_auto_backup_workspace(
    cursor,
    workspace_id: str,
    now: datetime,
    *,
    reason: str = "sync_push",
) -> dict[str, Any] | None:
    """
    Hourly automatic snapshot when prep docs changed. Never blocks sync if the
    backups table is missing (migration not applied yet).
    """
    try:
        cursor.execute(
            """
            SELECT created_at
            FROM workspace_backups
            WHERE workspace_id = %s AND source LIKE 'auto%%'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (workspace_id,),
        )
        latest = cursor.fetchone()
        if latest and latest["created_at"]:
            age = (now - latest["created_at"]).total_seconds()
            if age < _AUTO_BACKUP_MIN_INTERVAL_SECONDS:
                return None
        label = f"Auto {now.strftime('%Y-%m-%d %H:%M')} UTC ({reason})"
        return create_workspace_backup(
            cursor,
            workspace_id,
            label=label,
            source="auto_hourly",
            now=now,
        )
    except Exception:
        return None


def _prune_workspace_backups(cursor, workspace_id: str, source: str) -> None:
    keep = _AUTO_BACKUP_KEEP if source.startswith("auto") else _MANUAL_BACKUP_KEEP
    pattern = "auto%" if source.startswith("auto") else "manual%"
    cursor.execute(
        """
        DELETE FROM workspace_backups
        WHERE id IN (
            SELECT id FROM workspace_backups
            WHERE workspace_id = %s AND source LIKE %s
            ORDER BY created_at DESC
            OFFSET %s
        )
        """,
        (workspace_id, pattern, keep),
    )


def list_workspace_backups(cursor, workspace_id: str, *, limit: int = 50) -> list[dict]:
    cursor.execute(
        """
        SELECT id, label, source, created_at,
               pg_column_size(payload) AS bytes
        FROM workspace_backups
        WHERE workspace_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (workspace_id, min(limit, 200)),
    )
    return [
        {
            "id": row["id"],
            "label": row["label"],
            "source": row["source"],
            "createdAt": to_epoch_ms(row["created_at"]),
            "bytes": row["bytes"] or 0,
        }
        for row in cursor.fetchall()
    ]


def get_workspace_backup(cursor, workspace_id: str, backup_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT id, label, source, created_at, payload
        FROM workspace_backups
        WHERE workspace_id = %s AND id = %s
        """,
        (workspace_id, backup_id),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "label": row["label"],
        "source": row["source"],
        "createdAt": to_epoch_ms(row["created_at"]),
        "payload": row["payload"],
    }


def restore_workspace_backup(
    cursor,
    workspace_id: str,
    backup_id: int,
    now: datetime,
) -> dict[str, Any]:
    """
    Replace live library_records (and related rows present in the snapshot)
    from a backup. Archives the current workspace first as source=before_restore.
    """
    backup = get_workspace_backup(cursor, workspace_id, backup_id)
    if not backup:
        raise ValueError("backup_not_found")

    create_workspace_backup(
        cursor,
        workspace_id,
        label=f"Before restore of #{backup_id}",
        source="before_restore",
        now=now,
    )

    payload = backup["payload"] or {}
    records = _library_records_from_backup_payload(payload)
    # Restore library_records with a bumped updated_at so clients pull them.
    restored = 0
    for record in records:
        kind = record.get("kind")
        record_id = record.get("id")
        data = record.get("data")
        if not isinstance(kind, str) or not isinstance(record_id, str):
            continue
        if kind in _BACKUP_TRIGGER_KINDS or kind in {
            "opinions",
            "case_facts",
            "cites",
            "timeline",
            "note_tabs",
            "article_titles",
            "pdf_bookmarks",
        }:
            _archive_library_record_before_overwrite(
                cursor,
                workspace_id,
                {"kind": kind, "id": record_id, "data": data},
                now,
            )
            cursor.execute(
                """
                INSERT INTO library_records (workspace_id, kind, id, data, updated_at, deleted_at)
                VALUES (%s, %s, %s, %s, %s, NULL)
                ON CONFLICT (workspace_id, kind, id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = EXCLUDED.updated_at,
                    deleted_at = NULL
                """,
                (workspace_id, kind, record_id, Jsonb(data), now),
            )
            restored += 1

    return {
        "restoredBackupId": backup_id,
        "label": backup["label"],
        "libraryRecords": restored,
        "hasArguments": any(r.get("kind") == "arguments" for r in records),
    }


def _library_records_from_backup_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalize backup shapes into a list of {kind, id, data} rows.

    Supports v2 named keys (arguments / notebook / …), the v1 library_records
    array, and incomplete dict snapshots that only stored arguments.
    """
    if not isinstance(payload, dict):
        return []

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, record_id: str, data: Any) -> None:
        key = (kind, record_id)
        if key in seen:
            return
        seen.add(key)
        records.append({"kind": kind, "id": record_id, "data": data})

    raw = payload.get("library_records")
    if isinstance(raw, list):
        for record in raw:
            if not isinstance(record, dict):
                continue
            kind = record.get("kind")
            record_id = record.get("id")
            if isinstance(kind, str) and isinstance(record_id, str):
                add(kind, record_id, record.get("data"))
    elif isinstance(raw, dict):
        # Incomplete manual snapshots: {"arguments": board, ...}
        for kind, data in raw.items():
            if not isinstance(kind, str) or data is None:
                continue
            if isinstance(data, dict) and isinstance(data.get("id"), str) and "data" in data:
                add(kind, data["id"], data.get("data"))
            else:
                add(kind, "main", data)

    for kind in _BACKUP_DOC_KINDS:
        rows = payload.get(kind)
        if isinstance(rows, list):
            for record in rows:
                if isinstance(record, dict) and isinstance(record.get("id"), str):
                    add(kind, record["id"], record.get("data"))
        elif isinstance(rows, dict):
            # Single board object saved under the kind key.
            add(kind, str(rows.get("id") or "main"), rows)

    return records


def document_to_wire(row: dict[str, Any]) -> dict[str, Any]:
    """Shape a documents row the way the client stores it in filesMeta."""
    return {
        "id": row["id"],
        "caseId": row["case_id"],
        "name": row["name"],
        "size": row["size_bytes"],
        "contentType": row["content_type"],
        # True once the bytes are in Blob, which is what lets the UI stop
        # saying the PDF is browser-only.
        "stored": row["blob_pathname"] is not None,
        "updatedAt": to_epoch_ms(row["updated_at"]),
        "deleted": row.get("deleted_at") is not None,
    }


def upsert_document_blob(
    cursor,
    workspace_id: str,
    document_id: str,
    case_id: str,
    name: str,
    size_bytes: int,
    content_type: str,
    blob_pathname: str,
    blob_url: str,
    now: datetime,
) -> dict[str, Any]:
    """
    Record an uploaded PDF and where its bytes landed in Blob.

    This skips the last-write-wins guard that sync uses. The upload just
    happened, so it is by definition the newest state of that document.

    Returns:
        The stored document row in wire format.
    """
    cursor.execute(
        """
        INSERT INTO documents (
            workspace_id, id, case_id, name, size_bytes, content_type,
            blob_pathname, blob_url, updated_at, deleted_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL)
        ON CONFLICT (workspace_id, id) DO UPDATE SET
            case_id = EXCLUDED.case_id,
            name = EXCLUDED.name,
            size_bytes = EXCLUDED.size_bytes,
            content_type = EXCLUDED.content_type,
            blob_pathname = EXCLUDED.blob_pathname,
            blob_url = EXCLUDED.blob_url,
            updated_at = EXCLUDED.updated_at,
            deleted_at = NULL
        RETURNING id, case_id, name, size_bytes, content_type, blob_pathname,
                  updated_at, deleted_at
        """,
        (
            workspace_id,
            document_id,
            case_id,
            name,
            size_bytes,
            content_type,
            blob_pathname,
            blob_url,
            now,
        ),
    )
    return document_to_wire(cursor.fetchone())


def get_document(cursor, workspace_id: str, document_id: str) -> dict[str, Any] | None:
    """Fetch one live document row, or None when missing or already deleted."""
    cursor.execute(
        """
        SELECT id, case_id, name, size_bytes, content_type, blob_pathname,
               blob_url, updated_at
        FROM documents
        WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
        """,
        (workspace_id, document_id),
    )
    return cursor.fetchone()


def soft_delete_document(
    cursor,
    workspace_id: str,
    document_id: str,
    now: datetime,
) -> str | None:
    """
    Tombstone a document so other devices drop it too.

    Returns:
        The blob pathname that should now be deleted, or None when the document
        did not exist or held no bytes.
    """
    # Read the pathname before clearing it. RETURNING hands back the new row,
    # which would always be the NULL we just wrote.
    cursor.execute(
        "SELECT blob_pathname FROM documents WHERE workspace_id = %s AND id = %s",
        (workspace_id, document_id),
    )
    existing = cursor.fetchone()
    if existing is None:
        return None

    cursor.execute(
        """
        UPDATE documents
        SET deleted_at = %s, updated_at = %s, blob_pathname = NULL
        WHERE workspace_id = %s AND id = %s
        """,
        (now, now, workspace_id, document_id),
    )
    return existing["blob_pathname"]


def workspace_counts(cursor, workspace_id: str) -> dict[str, int]:
    """Live row counts per collection, for the sync status line in the UI."""
    counts: dict[str, int] = {}
    for spec in ENTITIES:
        cursor.execute(
            f"SELECT count(*) AS live FROM {spec.table} "
            "WHERE workspace_id = %s AND deleted_at IS NULL",
            (workspace_id,),
        )
        counts[spec.name] = cursor.fetchone()["live"]
    return counts

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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from psycopg.types.json import Jsonb

from app.db.notebook_merge import merge_notebook_snapshots

# A single push is capped so one bad client cannot send an unbounded statement.
# The whole seeded library is a few hundred rows, so this is generous.
MAX_ROWS_PER_ENTITY = 2_000

# Critical prep docs: auto-backup when any of these kinds change.
_BACKUP_TRIGGER_KINDS = frozenset(
    {"notebook", "arguments", "guide_edits", "facts", "openings"}
)
_AUTO_BACKUP_MIN_INTERVAL_SECONDS = 60 * 60  # one auto snapshot per hour
_AUTO_BACKUP_KEEP = 200  # ~8 days of hourly + headroom
_MANUAL_BACKUP_KEEP = 100


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


def _archive_library_record_before_overwrite(
    cursor,
    workspace_id: str,
    row: dict[str, Any],
    now: datetime,
) -> None:
    """
    Copy the current library_records.data into history when a push would change it.

    Sync itself stays last-write-wins. This archive exists so a wiped arguments
    board (or notebook / guide edits) can be pulled back without Neon PITR.
    """
    kind = row.get("kind")
    record_id = row.get("id")
    if not isinstance(kind, str) or not isinstance(record_id, str):
        return

    try:
        cursor.execute(
            """
            SELECT data, updated_at
            FROM library_records
            WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
            """,
            (workspace_id, kind, record_id),
        )
        existing = cursor.fetchone()
        if not existing:
            return

        incoming = row.get("data")
        if incoming is None:
            return
        # Skip archive when the payload is identical; avoids noise on heartbeat syncs.
        if existing["data"] == incoming:
            return

        cursor.execute(
            """
            INSERT INTO library_record_revisions (
                workspace_id, kind, record_id, data, row_updated_at, revised_at, source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                workspace_id,
                kind,
                record_id,
                Jsonb(existing["data"]),
                existing["updated_at"],
                now,
                "sync_push",
            ),
        )
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
    except Exception:
        # Never let history bookkeeping block a sync. Migration may not be
        # applied yet on a preview deploy; the upsert below still proceeds.
        return


def push_changes(
    cursor,
    workspace_id: str,
    payload: dict[str, Sequence[dict]],
    now: datetime,
) -> dict[str, int]:
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
        Collection name to the number of rows actually written. A row that lost
        the conflict does not count, which makes a stale client visible.
    """
    written: dict[str, int] = {}

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
            if name == "library_records":
                if prepared.get("kind") in _BACKUP_TRIGGER_KINDS:
                    touched_backup_kinds = True
                prepared = _protect_notebook_from_smaller_push(
                    cursor, workspace_id, prepared
                )
                _archive_library_record_before_overwrite(
                    cursor, workspace_id, prepared, now
                )
            cursor.execute(
                statement,
                _push_values(spec, workspace_id, prepared, now),
            )
            count += cursor.rowcount
        written[name] = count
        if name == "library_records" and touched_backup_kinds and count:
            maybe_auto_backup_workspace(cursor, workspace_id, now, reason="sync_push")

    return written


def _protect_notebook_from_smaller_push(
    cursor,
    workspace_id: str,
    prepared: dict[str, Any],
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
        cursor.execute(
            """
            SELECT data
            FROM library_records
            WHERE workspace_id = %s AND kind = 'notebook' AND id = %s
              AND deleted_at IS NULL
            """,
            (workspace_id, prepared.get("id")),
        )
        existing = cursor.fetchone()
        if not existing or not isinstance(existing.get("data"), dict):
            return prepared
        merged = merge_notebook_snapshots(incoming, existing["data"])
        if merged != incoming:
            return {**prepared, "data": merged}
    except Exception:
        return prepared
    return prepared


def collect_workspace_backup_payload(cursor, workspace_id: str) -> dict[str, Any]:
    """
    Build a downloadable / restorable snapshot of everything that must survive
    a hard refresh or a bad sync (not PDF bytes — those live in Blob).
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

    return {
        "version": 1,
        "workspaceId": workspace_id,
        "library_records": library_records,
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
    return {
        "id": row["id"],
        "label": row["label"],
        "source": row["source"],
        "createdAt": to_epoch_ms(row["created_at"]),
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
    # Restore library_records with a bumped updated_at so clients pull them.
    for record in payload.get("library_records") or []:
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

    return {
        "restoredBackupId": backup_id,
        "label": backup["label"],
        "libraryRecords": len(payload.get("library_records") or []),
    }


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

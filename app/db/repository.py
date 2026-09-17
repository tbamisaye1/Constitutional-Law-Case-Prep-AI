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

# A single push is capped so one bad client cannot send an unbounded statement.
# The whole seeded library is a few hundred rows, so this is generous.
MAX_ROWS_PER_ENTITY = 2_000


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
        ),
        json_fields=frozenset({"rects"}),
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
        cursor.execute(
            f"""
            SELECT {columns}, updated_at, deleted_at
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
        for row in rows[:MAX_ROWS_PER_ENTITY]:
            cursor.execute(statement, _push_values(spec, workspace_id, row, now))
            count += cursor.rowcount
        written[name] = count

    return written


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

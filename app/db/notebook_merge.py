"""
Merge OneNote-shaped notebook snapshots so a smaller sync push cannot erase
sections that still have pages on the server.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _index_tree(nodes: list | None, out: dict[str, dict] | None = None) -> dict[str, dict]:
    out = out if out is not None else {}
    for node in nodes or []:
        node_id = node.get("id")
        if isinstance(node_id, str):
            out[node_id] = node
        if node.get("children"):
            _index_tree(node["children"], out)
    return out


def _section_has_pages(pages_by_section: dict, section_id: str) -> bool:
    pages = pages_by_section.get(section_id)
    return isinstance(pages, list) and len(pages) > 0


def _merge_children(primary: list | None, secondary: list | None) -> list:
    primary = primary or []
    secondary = secondary or []
    by_id: dict[str, dict] = {}
    order: list[str] = []

    for node in primary:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        by_id[node_id] = deepcopy(node)
        order.append(node_id)

    for node in secondary:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        if node_id not in by_id:
            by_id[node_id] = deepcopy(node)
            order.append(node_id)
            continue
        existing = by_id[node_id]
        if (existing.get("kind") == "group" or node.get("kind") == "group") and (
            existing.get("children") is not None or node.get("children") is not None
        ):
            existing["children"] = _merge_children(
                existing.get("children") or [],
                node.get("children") or [],
            )

    return [by_id[node_id] for node_id in order]


def merge_notebook_snapshots(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """
    Prefer primary (usually the newer sync winner). Keep secondary-only sections
    that still carry pages so a wiped client cannot erase server notes.
    """
    primary_tree = primary.get("tree") or []
    secondary_tree = secondary.get("tree") or []
    primary_pages = primary.get("pagesBySection") or {}
    secondary_pages = secondary.get("pagesBySection") or {}

    merged_tree = _merge_children(primary_tree, secondary_tree)
    primary_ids = _index_tree(primary_tree)
    secondary_ids = _index_tree(secondary_tree)

    def prune(nodes: list | None) -> list:
        out = []
        for node in nodes or []:
            next_node = dict(node)
            if next_node.get("children") is not None:
                next_node["children"] = prune(next_node.get("children") or [])
            only_secondary = next_node.get("id") in secondary_ids and next_node.get("id") not in primary_ids
            if (
                only_secondary
                and next_node.get("kind") == "section"
                and not _section_has_pages(secondary_pages, next_node["id"])
            ):
                continue
            if only_secondary and next_node.get("kind") == "group" and not next_node.get("children"):
                continue
            out.append(next_node)
        return out

    tree = prune(merged_tree)
    kept_ids = _index_tree(tree)

    pages_by_section: dict[str, Any] = {**deepcopy(secondary_pages), **deepcopy(primary_pages)}
    for section_id, pages in secondary_pages.items():
        if section_id not in kept_ids:
            pages_by_section.pop(section_id, None)
            continue
        primary_list = primary_pages.get(section_id)
        if (not primary_list) and pages:
            pages_by_section[section_id] = deepcopy(pages)

    for section_id in list(pages_by_section.keys()):
        if section_id not in kept_ids:
            del pages_by_section[section_id]

    return {"tree": tree, "pagesBySection": pages_by_section}

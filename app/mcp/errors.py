"""Structured MCP tool errors."""

from __future__ import annotations

from typing import Any


class McpToolError(Exception):
    """Raised inside tool helpers; converted to the wire error shape."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


def error_result(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return McpToolError(code, message, details).as_dict()


def wrap_tool_result(value: Any) -> Any:
    """Pass through success payloads; convert McpToolError to wire shape."""
    if isinstance(value, McpToolError):
        return value.as_dict()
    return value

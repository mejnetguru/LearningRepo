#!/usr/bin/env python3
"""Command-line parser for the macOS Messages database.

This program reads Apple's ``chat.db`` SQLite database and exposes a small
command-line interface for inspecting chats, messages, and contacts.

Notes:
    The default database location is ``~/Library/Messages/chat.db``.
    The terminal application running this script usually needs Full Disk Access
    in macOS System Settings to read that file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, cast

APPLE_EPOCH_OFFSET_SECONDS: Final[int] = 978_307_200
APPLE_NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000

TextOrJson = Literal["text", "json"]
MessageOutput = Literal["text", "json", "csv"]


def apple_timestamp_to_datetime(value: int | float | None) -> datetime | None:
    """Convert an Apple Core Data timestamp to a timezone-aware UTC datetime.

    Messages timestamps are commonly stored as nanoseconds since
    ``2001-01-01 00:00:00 UTC``. Some rows may be missing or zero.

    Args:
        value: Raw timestamp value from SQLite.

    Returns:
        A timezone-aware UTC datetime, or ``None`` when the input is missing.
    """
    if value in (None, 0):
        return None

    unix_seconds = (
        float(value) / APPLE_NANOSECONDS_PER_SECOND
    ) + APPLE_EPOCH_OFFSET_SECONDS
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc)


def datetime_to_apple_timestamp(value: datetime) -> int:
    """Convert a timezone-aware datetime to an Apple Core Data timestamp."""
    normalized_value = value.astimezone(timezone.utc)
    unix_seconds = normalized_value.timestamp()
    return int(
        (unix_seconds - APPLE_EPOCH_OFFSET_SECONDS) * APPLE_NANOSECONDS_PER_SECOND
    )


def parse_iso_datetime(value: str, *, flag_name: str) -> datetime:
    """Parse an ISO-8601 date or datetime string from the command line.

    Naive values are treated as UTC.

    Args:
        value: Date or datetime string from the CLI.
        flag_name: Flag used to produce a precise error message.

    Returns:
        A timezone-aware datetime in UTC.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    try:
        parsed_value = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value for {flag_name}: {value!r}. "
            "Use ISO format such as 2026-03-30 or 2026-03-30T14:30:00."
        ) from exc

    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class Handle:
    """A contact handle stored in the Messages database."""

    row_id: int
    identifier: str
    country: str | None
    service: str | None


@dataclass(frozen=True, slots=True)
class Attachment:
    """A file attached to a message."""

    row_id: int
    guid: str
    filename: str | None
    mime_type: str | None
    transfer_state: int
    total_bytes: int

    @property
    def display_name(self) -> str:
        """Return a user-friendly filename."""
        if not self.filename:
            return "<unknown>"
        return Path(self.filename).name

    @property
    def size_kilobytes(self) -> float:
        """Return the attachment size in kilobytes."""
        return self.total_bytes / 1024


@dataclass(frozen=True, slots=True)
class Message:
    """A single message row plus related metadata."""

    row_id: int
    guid: str
    text: str | None
    handle_id: int | None
    sent_at: datetime | None
    read_at: datetime | None
    delivered_at: datetime | None
    is_from_me: bool
    is_read: bool
    service: str | None
    handle: Handle | None = None
    attachments: tuple[Attachment, ...] = ()

    @property
    def sender_label(self) -> str:
        """Return a human-readable sender name."""
        if self.is_from_me:
            return "Me"
        if self.handle is not None:
            return self.handle.identifier
        return "Unknown"


@dataclass(frozen=True, slots=True)
class Chat:
    """A conversation thread in the Messages database."""

    row_id: int
    guid: str
    chat_identifier: str
    display_name: str | None
    service_name: str | None
    is_archived: bool
    handles: tuple[Handle, ...] = ()

    @property
    def title(self) -> str:
        """Return the display name or a fallback identifier."""
        if self.display_name:
            return self.display_name
        return self.chat_identifier


@dataclass(frozen=True, slots=True)
class ChatsCommand:
    """Parsed arguments for the ``chats`` command."""

    include_archived: bool
    output: TextOrJson


@dataclass(frozen=True, slots=True)
class MessagesCommand:
    """Parsed arguments for the ``messages`` command."""

    chat_id: int | None
    handle_identifier: str | None
    search_text: str | None
    from_date: datetime | None
    to_date: datetime | None
    limit: int | None
    output: MessageOutput


@dataclass(frozen=True, slots=True)
class ContactsCommand:
    """Parsed arguments for the ``contacts`` command."""

    output: TextOrJson


Command = ChatsCommand | MessagesCommand | ContactsCommand


class MessagesDatabase:
    """Read-only interface to the macOS Messages SQLite database."""

    default_path: Final[Path] = Path.home() / "Library" / "Messages" / "chat.db"

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the database wrapper.

        Args:
            db_path: Optional custom path to ``chat.db``.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        self._db_path = db_path if db_path is not None else self.default_path
        if not self._db_path.exists():
            raise FileNotFoundError(
                "Messages database not found at "
                f"{self._db_path}. Grant Full Disk Access to your terminal if needed."
            )
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> MessagesDatabase:
        """Open the database connection."""
        database_uri = f"file:{self._db_path}?mode=ro"
        self._connection = sqlite3.connect(database_uri, uri=True)
        self._connection.row_factory = sqlite3.Row
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active connection.

        Raises:
            RuntimeError: If the database is not open.
        """
        if self._connection is None:
            raise RuntimeError("Database connection is not open.")
        return self._connection

    def get_handles(self) -> list[Handle]:
        """Return all known contact handles."""
        rows = self.connection.execute(
            """
            SELECT rowid, id, country, service
            FROM handle
            ORDER BY id
            """
        ).fetchall()

        return [
            Handle(
                row_id=self._require_int(row, "rowid"),
                identifier=self._require_str(row, "id"),
                country=self._optional_str(row, "country"),
                service=self._optional_str(row, "service"),
            )
            for row in rows
        ]

    def get_handle_map(self) -> dict[int, Handle]:
        """Return handles keyed by row id."""
        return {handle.row_id: handle for handle in self.get_handles()}

    def get_chats(self, *, include_archived: bool = False) -> list[Chat]:
        """Return chats with their participants attached."""
        conditions: list[str] = []
        parameters: list[object] = []
        if not include_archived:
            conditions.append("is_archived = ?")
            parameters.append(0)

        query_lines = [
            (
                "SELECT rowid, guid, chat_identifier, display_name, "
                "service_name, is_archived"
            ),
            "FROM chat",
        ]
        if conditions:
            query_lines.append("WHERE " + " AND ".join(conditions))
        query_lines.append("ORDER BY rowid")
        query = "\n".join(query_lines)

        rows = self.connection.execute(query, parameters).fetchall()
        handle_map = self.get_handle_map()

        chats: list[Chat] = []
        for row in rows:
            chat_row_id = self._require_int(row, "rowid")
            chats.append(
                Chat(
                    row_id=chat_row_id,
                    guid=self._require_str(row, "guid"),
                    chat_identifier=self._require_str(row, "chat_identifier"),
                    display_name=self._optional_str(row, "display_name"),
                    service_name=self._optional_str(row, "service_name"),
                    is_archived=bool(self._require_int(row, "is_archived")),
                    handles=self._get_handles_for_chat(chat_row_id, handle_map),
                )
            )

        return chats

    def get_messages(
        self,
        *,
        chat_id: int | None = None,
        handle_identifier: str | None = None,
        search_text: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Return messages matching the provided filters."""
        select_lines = [
            "SELECT DISTINCT",
            "    m.rowid,",
            "    m.guid,",
            "    m.text,",
            "    m.handle_id,",
            "    m.date,",
            "    m.date_read,",
            "    m.date_delivered,",
            "    m.is_from_me,",
            "    m.is_read,",
            "    m.service,",
            "    m.cache_has_attachments",
            "FROM message AS m",
        ]

        joins: list[str] = []
        conditions: list[str] = []
        parameters: list[object] = []

        if chat_id is not None:
            joins.append("JOIN chat_message_join AS cmj ON cmj.message_id = m.rowid")
            conditions.append("cmj.chat_id = ?")
            parameters.append(chat_id)

        if handle_identifier is not None:
            joins.append("JOIN handle AS h ON h.rowid = m.handle_id")
            conditions.append("h.id = ?")
            parameters.append(handle_identifier)

        if search_text is not None:
            conditions.append("m.text LIKE ?")
            parameters.append(f"%{search_text}%")

        if from_date is not None:
            conditions.append("m.date >= ?")
            parameters.append(datetime_to_apple_timestamp(from_date))

        if to_date is not None:
            conditions.append("m.date <= ?")
            parameters.append(datetime_to_apple_timestamp(to_date))

        query_lines = [*select_lines, *joins]
        if conditions:
            query_lines.append("WHERE " + " AND ".join(conditions))
        query_lines.append("ORDER BY m.date ASC")
        if limit is not None:
            query_lines.append("LIMIT ?")
            parameters.append(limit)

        query = "\n".join(query_lines)

        rows = self.connection.execute(query, parameters).fetchall()
        handle_map = self.get_handle_map()

        messages: list[Message] = []
        for row in rows:
            handle_id = self._optional_int(row, "handle_id")
            resolved_handle = handle_map.get(handle_id) if handle_id is not None else None
            row_id = self._require_int(row, "rowid")
            has_attachments = bool(self._require_int(row, "cache_has_attachments"))
            attachments = self._get_attachments_for_message(row_id) if has_attachments else ()

            messages.append(
                Message(
                    row_id=row_id,
                    guid=self._require_str(row, "guid"),
                    text=self._optional_str(row, "text"),
                    handle_id=handle_id,
                    sent_at=apple_timestamp_to_datetime(
                        self._optional_number(row, "date")
                    ),
                    read_at=apple_timestamp_to_datetime(
                        self._optional_number(row, "date_read")
                    ),
                    delivered_at=apple_timestamp_to_datetime(
                        self._optional_number(row, "date_delivered")
                    ),
                    is_from_me=bool(self._require_int(row, "is_from_me")),
                    is_read=bool(self._require_int(row, "is_read")),
                    service=self._optional_str(row, "service"),
                    handle=resolved_handle,
                    attachments=attachments,
                )
            )

        return messages

    def _get_handles_for_chat(
        self,
        chat_id: int,
        handle_map: dict[int, Handle],
    ) -> tuple[Handle, ...]:
        """Return participants for a chat."""
        rows = self.connection.execute(
            """
            SELECT handle_id
            FROM chat_handle_join
            WHERE chat_id = ?
            ORDER BY handle_id
            """,
            (chat_id,),
        ).fetchall()

        handles = [
            handle_map[handle_id]
            for row in rows
            for handle_id in [self._require_int(row, "handle_id")]
            if handle_id in handle_map
        ]
        return tuple(handles)

    def _get_attachments_for_message(self, message_id: int) -> tuple[Attachment, ...]:
        """Return attachments linked to a message."""
        rows = self.connection.execute(
            """
            SELECT a.rowid, a.guid, a.filename, a.mime_type,
                   a.transfer_state, a.total_bytes
            FROM attachment AS a
            JOIN message_attachment_join AS maj ON maj.attachment_id = a.rowid
            WHERE maj.message_id = ?
            ORDER BY a.rowid
            """,
            (message_id,),
        ).fetchall()

        attachments = [
            Attachment(
                row_id=self._require_int(row, "rowid"),
                guid=self._require_str(row, "guid"),
                filename=self._optional_str(row, "filename"),
                mime_type=self._optional_str(row, "mime_type"),
                transfer_state=self._optional_int(row, "transfer_state") or 0,
                total_bytes=self._optional_int(row, "total_bytes") or 0,
            )
            for row in rows
        ]
        return tuple(attachments)

    @staticmethod
    def _require_int(row: sqlite3.Row, key: str) -> int:
        """Read a required integer column from a SQLite row."""
        value = row[key]
        if not isinstance(value, int):
            raise TypeError(f"Expected integer value for column {key!r}.")
        return value

    @staticmethod
    def _optional_int(row: sqlite3.Row, key: str) -> int | None:
        """Read an optional integer column from a SQLite row."""
        value = row[key]
        if value is None:
            return None
        if not isinstance(value, int):
            raise TypeError(f"Expected integer or null value for column {key!r}.")
        return value

    @staticmethod
    def _optional_number(row: sqlite3.Row, key: str) -> int | float | None:
        """Read an optional numeric column from a SQLite row."""
        value = row[key]
        if value is None:
            return None
        if not isinstance(value, int | float):
            raise TypeError(f"Expected numeric or null value for column {key!r}.")
        return cast(int | float, value)

    @staticmethod
    def _require_str(row: sqlite3.Row, key: str) -> str:
        """Read a required string column from a SQLite row."""
        value = row[key]
        if not isinstance(value, str):
            raise TypeError(f"Expected string value for column {key!r}.")
        return value

    @staticmethod
    def _optional_str(row: sqlite3.Row, key: str) -> str | None:
        """Read an optional string column from a SQLite row."""
        value = row[key]
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"Expected string or null value for column {key!r}.")
        return value


class OutputFormatter:
    """Format models as text, JSON, or CSV output."""

    def format_chats(self, chats: list[Chat], output: TextOrJson) -> str:
        """Format chats in the requested output format."""
        if output == "json":
            return self._format_chats_json(chats)
        return self._format_chats_text(chats)

    def format_messages(self, messages: list[Message], output: MessageOutput) -> str:
        """Format messages in the requested output format."""
        if output == "json":
            return self._format_messages_json(messages)
        if output == "csv":
            return self._format_messages_csv(messages)
        return self._format_messages_text(messages)

    def format_contacts(self, handles: list[Handle], output: TextOrJson) -> str:
        """Format contacts in the requested output format."""
        if output == "json":
            return self._format_contacts_json(handles)
        return self._format_contacts_text(handles)

    def _format_chats_text(self, chats: list[Chat]) -> str:
        if not chats:
            return "No chats found."

        lines: list[str] = []
        for chat in chats:
            participants = ", ".join(handle.identifier for handle in chat.handles)
            if not participants:
                participants = "(none)"
            archived_marker = " [archived]" if chat.is_archived else ""
            service_name = chat.service_name or "Unknown"
            lines.append(f"[{chat.row_id}] {chat.title}{archived_marker}")
            lines.append(f"  Service: {service_name}")
            lines.append(f"  Participants: {participants}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_chats_json(self, chats: list[Chat]) -> str:
        payload = [
            {
                "row_id": chat.row_id,
                "guid": chat.guid,
                "title": chat.title,
                "chat_identifier": chat.chat_identifier,
                "service_name": chat.service_name,
                "is_archived": chat.is_archived,
                "participants": [handle.identifier for handle in chat.handles],
            }
            for chat in chats
        ]
        return json.dumps(payload, indent=2)

    def _format_messages_text(self, messages: list[Message]) -> str:
        if not messages:
            return "No messages found."

        lines: list[str] = []
        for message in messages:
            sent_at = (
                message.sent_at.isoformat()
                if message.sent_at is not None
                else "<unknown>"
            )
            text = message.text if message.text else "<no text>"
            attachment_names = ", ".join(
                attachment.display_name for attachment in message.attachments
            )
            suffix = f" [{attachment_names}]" if attachment_names else ""
            lines.append(f"[{sent_at}] {message.sender_label}: {text}{suffix}")
        return "\n".join(lines)

    def _format_messages_json(self, messages: list[Message]) -> str:
        payload = [
            {
                "row_id": message.row_id,
                "guid": message.guid,
                "sent_at": message.sent_at.isoformat() if message.sent_at else None,
                "read_at": message.read_at.isoformat() if message.read_at else None,
                "delivered_at": (
                    message.delivered_at.isoformat() if message.delivered_at else None
                ),
                "sender": message.sender_label,
                "text": message.text,
                "service": message.service,
                "is_from_me": message.is_from_me,
                "is_read": message.is_read,
                "attachments": [
                    {
                        "row_id": attachment.row_id,
                        "filename": attachment.filename,
                        "display_name": attachment.display_name,
                        "mime_type": attachment.mime_type,
                        "size_kilobytes": round(attachment.size_kilobytes, 2),
                    }
                    for attachment in message.attachments
                ],
            }
            for message in messages
        ]
        return json.dumps(payload, indent=2)

    def _format_messages_csv(self, messages: list[Message]) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "row_id",
                "sent_at",
                "sender",
                "text",
                "service",
                "is_from_me",
                "is_read",
                "attachments",
            ],
        )
        writer.writeheader()
        for message in messages:
            writer.writerow(
                {
                    "row_id": message.row_id,
                    "sent_at": message.sent_at.isoformat() if message.sent_at else "",
                    "sender": message.sender_label,
                    "text": message.text or "",
                    "service": message.service or "",
                    "is_from_me": message.is_from_me,
                    "is_read": message.is_read,
                    "attachments": "; ".join(
                        attachment.display_name for attachment in message.attachments
                    ),
                }
            )
        return buffer.getvalue()

    def _format_contacts_text(self, handles: list[Handle]) -> str:
        if not handles:
            return "No contacts found."

        lines = [f"{'Row ID':>6}  {'Service':<10}  Identifier", "-" * 50]
        for handle in handles:
            lines.append(
                f"{handle.row_id:>6}  {(handle.service or 'Unknown'):<10}  {handle.identifier}"
            )
        return "\n".join(lines)

    def _format_contacts_json(self, handles: list[Handle]) -> str:
        payload = [
            {
                "row_id": handle.row_id,
                "identifier": handle.identifier,
                "country": handle.country,
                "service": handle.service,
            }
            for handle in handles
        ]
        return json.dumps(payload, indent=2)


class MessagesApplication:
    """Top-level CLI application."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the application state."""
        self._db_path = db_path
        self._formatter = OutputFormatter()

    def run(self, command: Command) -> int:
        """Execute a parsed command and print its output."""
        try:
            with MessagesDatabase(self._db_path) as database:
                if isinstance(command, ChatsCommand):
                    output = self._formatter.format_chats(
                        database.get_chats(include_archived=command.include_archived),
                        command.output,
                    )
                elif isinstance(command, MessagesCommand):
                    output = self._formatter.format_messages(
                        database.get_messages(
                            chat_id=command.chat_id,
                            handle_identifier=command.handle_identifier,
                            search_text=command.search_text,
                            from_date=command.from_date,
                            to_date=command.to_date,
                            limit=command.limit,
                        ),
                        command.output,
                    )
                else:
                    output = self._formatter.format_contacts(
                        database.get_handles(),
                        command.output,
                    )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except sqlite3.Error as exc:
            print(f"Database error: {exc}", file=sys.stderr)
            return 1

        print(
            output,
            end=""
            if isinstance(command, MessagesCommand) and command.output == "csv"
            else "\n",
        )
        return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="messages_parser",
        description="Inspect data from the macOS Messages chat.db database.",
        epilog=(
            "Examples:\n"
            "  %(prog)s chats\n"
            "  %(prog)s chats --include-archived --output json\n"
            "  %(prog)s messages --chat-id 5 --limit 25\n"
            "  %(prog)s messages --handle +15551234567 --output csv\n"
            "  %(prog)s messages --search hello --from-date 2026-03-01\n"
            "  %(prog)s contacts --output json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to chat.db. Defaults to ~/Library/Messages/chat.db.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    chats_parser = subparsers.add_parser("chats", help="List conversation threads.")
    chats_parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived chats in the output.",
    )
    chats_parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Choose text or json output.",
    )

    messages_parser = subparsers.add_parser("messages", help="List or search messages.")
    messages_parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Filter by chat row id.",
    )
    messages_parser.add_argument(
        "--handle",
        default=None,
        help="Filter by contact phone number or email address.",
    )
    messages_parser.add_argument(
        "--search",
        default=None,
        help="Filter messages whose text contains this value.",
    )
    messages_parser.add_argument(
        "--from-date",
        default=None,
        help="Earliest date in ISO format, such as 2026-03-01.",
    )
    messages_parser.add_argument(
        "--to-date",
        default=None,
        help="Latest date in ISO format, such as 2026-03-31T23:59:59.",
    )
    messages_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of messages to return.",
    )
    messages_parser.add_argument(
        "--output",
        choices=("text", "json", "csv"),
        default="text",
        help="Choose text, json, or csv output.",
    )

    contacts_parser = subparsers.add_parser("contacts", help="List stored contacts.")
    contacts_parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Choose text or json output.",
    )

    return parser


def namespace_to_command(arguments: argparse.Namespace) -> Command:
    """Convert ``argparse`` output into strongly typed command objects."""
    command_name = cast(str, arguments.command)
    if command_name == "chats":
        return ChatsCommand(
            include_archived=cast(bool, arguments.include_archived),
            output=cast(TextOrJson, arguments.output),
        )

    if command_name == "messages":
        limit = cast(int | None, arguments.limit)
        if limit is not None and limit <= 0:
            raise ValueError("--limit must be greater than zero.")

        from_date_text = cast(str | None, arguments.from_date)
        to_date_text = cast(str | None, arguments.to_date)
        from_date = (
            parse_iso_datetime(from_date_text, flag_name="--from-date")
            if from_date_text is not None
            else None
        )
        to_date = (
            parse_iso_datetime(to_date_text, flag_name="--to-date")
            if to_date_text is not None
            else None
        )
        if from_date is not None and to_date is not None and from_date > to_date:
            raise ValueError("--from-date cannot be later than --to-date.")

        return MessagesCommand(
            chat_id=cast(int | None, arguments.chat_id),
            handle_identifier=cast(str | None, arguments.handle),
            search_text=cast(str | None, arguments.search),
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            output=cast(MessageOutput, arguments.output),
        )

    if command_name == "contacts":
        return ContactsCommand(output=cast(TextOrJson, arguments.output))

    raise ValueError(f"Unsupported command: {command_name!r}")


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and run the application."""
    parser = build_argument_parser()

    try:
        parsed_args = parser.parse_args(argv)
        command = namespace_to_command(parsed_args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    application = MessagesApplication(db_path=cast(Path | None, parsed_args.db))
    return application.run(command)


if __name__ == "__main__":
    raise SystemExit(main())

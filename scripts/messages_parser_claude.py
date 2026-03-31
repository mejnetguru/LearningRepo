#!/usr/bin/env python3
"""MacOS Messages Parser.

Parses the MacOS Messages SQLite database (chat.db) to extract
conversations, messages, and contact information via a command-line interface.

Usage:
    python messages_parser.py chats
    python messages_parser.py messages --chat-id 5 --output json
    python messages_parser.py contacts

Note:
    Full Disk Access must be granted to the terminal application in
    System Settings > Privacy & Security > Full Disk Access.
"""

import argparse
import csv
import io
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Apple's Core Data reference date is January 1, 2001 00:00:00 UTC.
# This offset converts Apple timestamps to Unix timestamps.
_APPLE_EPOCH_OFFSET: int = 978_307_200  # seconds

# Messages stores dates as nanoseconds since the Apple epoch.
_NS_PER_SECOND: int = 1_000_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apple_ts_to_datetime(timestamp: int) -> datetime:
    """Convert an Apple Core Data nanosecond timestamp to a UTC datetime.

    Args:
        timestamp: Nanoseconds since 2001-01-01 00:00:00 UTC.

    Returns:
        A timezone-aware datetime in UTC.
    """
    if timestamp == 0:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    unix_ts = (timestamp / _NS_PER_SECOND) + _APPLE_EPOCH_OFFSET
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


def _datetime_to_apple_ts(dt: datetime) -> int:
    """Convert a UTC datetime to an Apple Core Data nanosecond timestamp.

    Args:
        dt: A timezone-aware datetime.

    Returns:
        Nanoseconds since 2001-01-01 00:00:00 UTC.
    """
    return int((dt.timestamp() - _APPLE_EPOCH_OFFSET) * _NS_PER_SECOND)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Handle:
    """A contact (phone number or email) stored in the Messages database."""

    rowid: int
    identifier: str  # phone number or email address
    country: str | None
    service: str  # "iMessage" or "SMS"

    def __str__(self) -> str:
        return self.identifier


@dataclass
class Attachment:
    """A file attached to a message."""

    rowid: int
    guid: str
    filename: str | None  # full path on disk (may use ~ tilde expansion)
    mime_type: str | None
    transfer_state: int
    total_bytes: int

    @property
    def display_name(self) -> str:
        """Return the base filename, or a placeholder if unknown."""
        if self.filename:
            return Path(self.filename).name
        return "<unknown>"

    @property
    def size_kb(self) -> float:
        """Return file size in kilobytes."""
        return self.total_bytes / 1024


@dataclass
class Message:
    """A single message in a conversation."""

    rowid: int
    guid: str
    text: str | None
    handle_id: int | None
    date: datetime
    date_read: datetime | None
    date_delivered: datetime | None
    is_from_me: bool
    is_read: bool
    service: str
    has_attachments: bool
    # Populated after construction via join queries
    handle: Handle | None = None
    attachments: list[Attachment] = field(default_factory=list)

    @property
    def sender(self) -> str:
        """Return a human-readable sender label."""
        if self.is_from_me:
            return "Me"
        return self.handle.identifier if self.handle else "Unknown"

    def __str__(self) -> str:
        ts = self.date.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        body = self.text or "<no text>"
        attachments = ""
        if self.attachments:
            names = ", ".join(a.display_name for a in self.attachments)
            attachments = f" [{names}]"
        return f"[{ts}] {self.sender}: {body}{attachments}"


@dataclass
class Chat:
    """A conversation thread (1-on-1 or group)."""

    rowid: int
    guid: str
    chat_identifier: str  # e.g. "+15551234567" or group UUID
    display_name: str | None  # set for group chats
    service_name: str
    is_archived: bool
    handles: list[Handle] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Return the display name, falling back to the chat identifier."""
        return self.display_name or self.chat_identifier

    def __str__(self) -> str:
        participants = ", ".join(str(h) for h in self.handles)
        archived = " [archived]" if self.is_archived else ""
        return f"[{self.rowid}] {self.name} ({self.service_name}){archived} — {participants}"


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------


class MessagesDatabase:
    """Manages a read-only connection to the MacOS Messages SQLite database.

    Intended for use as a context manager::

        with MessagesDatabase() as db:
            chats = db.get_chats()
    """

    DEFAULT_PATH: Path = Path.home() / "Library" / "Messages" / "chat.db"

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialise the database wrapper.

        Args:
            db_path: Path to chat.db.  Defaults to the standard MacOS location.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
        """
        self.db_path: Path = db_path or self.DEFAULT_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Messages database not found: {self.db_path}\n"
                "Tip: grant Full Disk Access to your terminal in\n"
                "     System Settings > Privacy & Security > Full Disk Access."
            )
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "MessagesDatabase":
        # Open in read-only URI mode to avoid accidental writes.
        uri = f"file:{self.db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, *_: object) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def _connection(self) -> sqlite3.Connection:
        """Return the active connection, raising if not open."""
        if self._conn is None:
            raise RuntimeError("Not connected. Use MessagesDatabase as a context manager.")
        return self._conn

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_handles(self) -> list[Handle]:
        """Return all handles (contacts) in the database."""
        rows = self._connection.execute(
            "SELECT rowid, id, country, service FROM handle ORDER BY id"
        ).fetchall()
        return [
            Handle(
                rowid=r["rowid"],
                identifier=r["id"],
                country=r["country"],
                service=r["service"],
            )
            for r in rows
        ]

    def get_handle_map(self) -> dict[int, Handle]:
        """Return a mapping of handle rowid → Handle."""
        return {h.rowid: h for h in self.get_handles()}

    def get_chats(self, *, include_archived: bool = False) -> list[Chat]:
        """Return all chats, optionally including archived ones.

        Args:
            include_archived: When True, archived chats are included.

        Returns:
            List of Chat objects with handles pre-populated.
        """
        where = "" if include_archived else "WHERE is_archived = 0"
        rows = self._connection.execute(
            f"""
            SELECT rowid, guid, chat_identifier, display_name,
                   service_name, is_archived
            FROM   chat
            {where}
            ORDER  BY rowid
            """
        ).fetchall()

        chats = [
            Chat(
                rowid=r["rowid"],
                guid=r["guid"],
                chat_identifier=r["chat_identifier"],
                display_name=r["display_name"],
                service_name=r["service_name"],
                is_archived=bool(r["is_archived"]),
            )
            for r in rows
        ]

        handle_map = self.get_handle_map()
        for chat in chats:
            chat.handles = self._get_handles_for_chat(chat.rowid, handle_map)

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
        """Return messages matching the given filters.

        All filters are optional and combined with AND logic.

        Args:
            chat_id: Restrict to this chat's rowid.
            handle_identifier: Restrict to messages from/to this contact
                (phone number or email).
            search_text: Case-insensitive substring match against message text.
            from_date: Only messages on or after this UTC datetime.
            to_date: Only messages on or before this UTC datetime.
            limit: Maximum number of messages to return (most recent N if used
                with ordering).

        Returns:
            Messages sorted by date ascending.
        """
        handle_map = self.get_handle_map()

        # Build the SELECT with optional JOINs to avoid cross joins.
        select = """
            SELECT DISTINCT
                m.rowid, m.guid, m.text, m.handle_id,
                m.date, m.date_read, m.date_delivered,
                m.is_from_me, m.is_read, m.service,
                m.cache_has_attachments
            FROM message m
        """
        joins: list[str] = []
        conditions: list[str] = []
        params: list[int | str] = []

        if chat_id is not None:
            joins.append("JOIN chat_message_join cmj ON m.rowid = cmj.message_id")
            conditions.append("cmj.chat_id = ?")
            params.append(chat_id)

        if handle_identifier is not None:
            joins.append("JOIN handle h ON m.handle_id = h.rowid")
            conditions.append("h.id = ?")
            params.append(handle_identifier)

        if search_text is not None:
            conditions.append("m.text LIKE ?")
            params.append(f"%{search_text}%")

        if from_date is not None:
            conditions.append("m.date >= ?")
            params.append(_datetime_to_apple_ts(from_date))

        if to_date is not None:
            conditions.append("m.date <= ?")
            params.append(_datetime_to_apple_ts(to_date))

        query = select
        if joins:
            query += " " + " ".join(joins)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY m.date ASC"
        if limit is not None:
            query += f" LIMIT {limit}"

        rows = self._connection.execute(query, params).fetchall()

        messages: list[Message] = []
        for r in rows:
            msg = Message(
                rowid=r["rowid"],
                guid=r["guid"],
                text=r["text"],
                handle_id=r["handle_id"],
                date=_apple_ts_to_datetime(r["date"] or 0),
                date_read=_apple_ts_to_datetime(r["date_read"]) if r["date_read"] else None,
                date_delivered=(
                    _apple_ts_to_datetime(r["date_delivered"]) if r["date_delivered"] else None
                ),
                is_from_me=bool(r["is_from_me"]),
                is_read=bool(r["is_read"]),
                service=r["service"] or "",
                has_attachments=bool(r["cache_has_attachments"]),
            )
            if r["handle_id"] and r["handle_id"] in handle_map:
                msg.handle = handle_map[r["handle_id"]]
            if msg.has_attachments:
                msg.attachments = self._get_attachments_for_message(msg.rowid)
            messages.append(msg)

        return messages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_handles_for_chat(
        self, chat_id: int, handle_map: dict[int, Handle]
    ) -> list[Handle]:
        """Return the handles participating in a given chat."""
        rows = self._connection.execute(
            "SELECT handle_id FROM chat_handle_join WHERE chat_id = ?",
            (chat_id,),
        ).fetchall()
        return [handle_map[r["handle_id"]] for r in rows if r["handle_id"] in handle_map]

    def _get_attachments_for_message(self, message_id: int) -> list[Attachment]:
        """Return all attachments linked to a message."""
        rows = self._connection.execute(
            """
            SELECT a.rowid, a.guid, a.filename, a.mime_type,
                   a.transfer_state, a.total_bytes
            FROM   attachment a
            JOIN   message_attachment_join maj ON a.rowid = maj.attachment_id
            WHERE  maj.message_id = ?
            """,
            (message_id,),
        ).fetchall()
        return [
            Attachment(
                rowid=r["rowid"],
                guid=r["guid"],
                filename=r["filename"],
                mime_type=r["mime_type"],
                transfer_state=r["transfer_state"] or 0,
                total_bytes=r["total_bytes"] or 0,
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class OutputFormatter:
    """Serialises data models to text, JSON, or CSV strings."""

    # ------------------------------------------------------------------
    # Chats
    # ------------------------------------------------------------------

    @staticmethod
    def chats_as_text(chats: list[Chat]) -> str:
        """Format a list of chats as a human-readable table."""
        if not chats:
            return "No chats found."
        lines: list[str] = []
        for chat in chats:
            participants = ", ".join(str(h) for h in chat.handles) or "(none)"
            archived = "  [archived]" if chat.is_archived else ""
            lines.append(f"[{chat.rowid:>4}]  {chat.name}{archived}")
            lines.append(f"         Service: {chat.service_name}")
            lines.append(f"    Participants: {participants}")
            lines.append("")
        return "\n".join(lines).rstrip()

    @staticmethod
    def chats_as_json(chats: list[Chat]) -> str:
        """Serialise a list of chats to a JSON string."""
        data = [
            {
                "rowid": c.rowid,
                "name": c.name,
                "identifier": c.chat_identifier,
                "service": c.service_name,
                "is_archived": c.is_archived,
                "participants": [h.identifier for h in c.handles],
            }
            for c in chats
        ]
        return json.dumps(data, indent=2)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    @staticmethod
    def messages_as_text(messages: list[Message]) -> str:
        """Format messages as a human-readable transcript."""
        if not messages:
            return "No messages found."
        return "\n".join(str(m) for m in messages)

    @staticmethod
    def messages_as_json(messages: list[Message]) -> str:
        """Serialise messages to a JSON string."""
        data = [
            {
                "rowid": m.rowid,
                "guid": m.guid,
                "date": m.date.isoformat(),
                "sender": m.sender,
                "text": m.text,
                "is_read": m.is_read,
                "service": m.service,
                "attachments": [
                    {
                        "name": a.display_name,
                        "mime_type": a.mime_type,
                        "size_kb": round(a.size_kb, 2),
                    }
                    for a in m.attachments
                ],
            }
            for m in messages
        ]
        return json.dumps(data, indent=2)

    @staticmethod
    def messages_as_csv(messages: list[Message]) -> str:
        """Serialise messages to a CSV string."""
        buf = io.StringIO()
        fieldnames = ["date", "sender", "text", "service", "is_read", "attachments"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for m in messages:
            writer.writerow(
                {
                    "date": m.date.isoformat(),
                    "sender": m.sender,
                    "text": m.text or "",
                    "service": m.service,
                    "is_read": m.is_read,
                    "attachments": "; ".join(a.display_name for a in m.attachments),
                }
            )
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Contacts / handles
    # ------------------------------------------------------------------

    @staticmethod
    def handles_as_text(handles: list[Handle]) -> str:
        """Format handles as a human-readable table."""
        if not handles:
            return "No contacts found."
        lines = [f"{'ID':>6}  {'Service':<10}  Identifier", "-" * 50]
        for h in handles:
            lines.append(f"{h.rowid:>6}  {h.service:<10}  {h.identifier}")
        return "\n".join(lines)

    @staticmethod
    def handles_as_json(handles: list[Handle]) -> str:
        """Serialise handles to a JSON string."""
        data = [
            {
                "rowid": h.rowid,
                "identifier": h.identifier,
                "service": h.service,
                "country": h.country,
            }
            for h in handles
        ]
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# CLI application
# ---------------------------------------------------------------------------


class MessagesParser:
    """Top-level CLI application for querying the MacOS Messages database."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialise the application.

        Args:
            db_path: Optional override for the chat.db location.
        """
        self.db_path = db_path
        self.fmt = OutputFormatter()

    def run(self, args: argparse.Namespace) -> int:
        """Dispatch to the appropriate subcommand handler.

        Args:
            args: Parsed argument namespace from :func:`build_arg_parser`.

        Returns:
            POSIX exit code (0 = success).
        """
        try:
            with MessagesDatabase(self.db_path) as db:
                match args.command:
                    case "chats":
                        return self._cmd_chats(db, args)
                    case "messages":
                        return self._cmd_messages(db, args)
                    case "contacts":
                        return self._cmd_contacts(db, args)
                    case _:
                        print(f"Unknown command: {args.command!r}", file=sys.stderr)
                        return 1
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except sqlite3.OperationalError as exc:
            print(f"Database error: {exc}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Subcommand handlers
    # ------------------------------------------------------------------

    def _cmd_chats(self, db: MessagesDatabase, args: argparse.Namespace) -> int:
        """List all conversations."""
        chats = db.get_chats(include_archived=args.include_archived)
        if args.output == "json":
            print(self.fmt.chats_as_json(chats))
        else:
            print(self.fmt.chats_as_text(chats))
        return 0

    def _cmd_messages(self, db: MessagesDatabase, args: argparse.Namespace) -> int:
        """Fetch and display messages with optional filters."""
        from_date = self._parse_date(args.from_date, "--from-date")
        if from_date is None and args.from_date:
            return 1  # parse error already printed

        to_date = self._parse_date(args.to_date, "--to-date")
        if to_date is None and args.to_date:
            return 1

        messages = db.get_messages(
            chat_id=args.chat_id,
            handle_identifier=args.handle,
            search_text=args.search,
            from_date=from_date,
            to_date=to_date,
            limit=args.limit,
        )

        match args.output:
            case "json":
                print(self.fmt.messages_as_json(messages))
            case "csv":
                print(self.fmt.messages_as_csv(messages), end="")
            case _:
                print(self.fmt.messages_as_text(messages))
        return 0

    def _cmd_contacts(self, db: MessagesDatabase, args: argparse.Namespace) -> int:
        """List all contacts (handles)."""
        handles = db.get_handles()
        if args.output == "json":
            print(self.fmt.handles_as_json(handles))
        else:
            print(self.fmt.handles_as_text(handles))
        return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: str | None, flag: str) -> datetime | None:
        """Parse an ISO date string supplied on the command line.

        Args:
            value: Raw string from argparse (e.g. "2024-01-01").
            flag: The CLI flag name, used in error messages.

        Returns:
            A timezone-aware UTC datetime, or None if *value* is falsy.
            Prints an error and returns None if the string is malformed.
        """
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
            # Treat naive dates as UTC midnight.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            print(
                f"Invalid {flag} value {value!r}. "
                "Expected ISO format, e.g. 2024-01-15 or 2024-01-15T09:00:00",
                file=sys.stderr,
            )
            return None


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="messages_parser",
        description="Query MacOS Messages data from ~/Library/Messages/chat.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s chats
  %(prog)s chats --include-archived --output json
  %(prog)s messages --chat-id 5 --limit 20
  %(prog)s messages --handle +15551234567 --output csv
  %(prog)s messages --search "hello" --from-date 2024-01-01
  %(prog)s contacts --output json
        """,
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        type=Path,
        default=None,
        help="Path to chat.db (default: ~/Library/Messages/chat.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ---- chats -----------------------------------------------------------
    p_chats = sub.add_parser("chats", help="List all conversations")
    p_chats.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived chats",
    )
    p_chats.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        metavar="FORMAT",
        help="Output format: text (default) or json",
    )

    # ---- messages --------------------------------------------------------
    p_msgs = sub.add_parser("messages", help="List and search messages")
    p_msgs.add_argument(
        "--chat-id",
        type=int,
        default=None,
        metavar="ID",
        help="Filter by chat rowid (see 'chats' output)",
    )
    p_msgs.add_argument(
        "--handle",
        default=None,
        metavar="CONTACT",
        help="Filter by contact phone number or email address",
    )
    p_msgs.add_argument(
        "--search",
        default=None,
        metavar="TEXT",
        help="Return only messages whose text contains TEXT",
    )
    p_msgs.add_argument(
        "--from-date",
        default=None,
        metavar="DATE",
        help="Earliest message date, ISO format (e.g. 2024-01-01)",
    )
    p_msgs.add_argument(
        "--to-date",
        default=None,
        metavar="DATE",
        help="Latest message date, ISO format (e.g. 2024-12-31)",
    )
    p_msgs.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of messages to return",
    )
    p_msgs.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        metavar="FORMAT",
        help="Output format: text (default), json, or csv",
    )

    # ---- contacts --------------------------------------------------------
    p_contacts = sub.add_parser("contacts", help="List all contacts (handles)")
    p_contacts.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        metavar="FORMAT",
        help="Output format: text (default) or json",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments and run the application."""
    parser = build_arg_parser()
    args = parser.parse_args()
    app = MessagesParser(db_path=args.db)
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())

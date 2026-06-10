import html
import os
from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo import MongoClient


DATABASE_NAME = "health_tracker"
DEFAULT_USER_ID = "default"


_client: MongoClient | None = None
_db = None


def _get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(os.environ["MONGO_URI"])
        _db = _client[DATABASE_NAME]
    return _db


def _utc_now() -> datetime:
    return datetime.utcnow()


def _date_str(value: datetime | None = None) -> str:
    return (value or _utc_now()).strftime("%Y-%m-%d")


def _serialize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(doc)
    if isinstance(serialized.get("_id"), ObjectId):
        serialized["_id"] = str(serialized["_id"])
    return serialized


def _serialize_docs(docs) -> list[dict]:
    return [_serialize_doc(doc) for doc in docs]


def save_raw_entry(raw_text, timestamp=None) -> str:
    timestamp = timestamp or _utc_now()
    doc = {
        "user_id": DEFAULT_USER_ID,
        "raw_text": raw_text,
        "timestamp": timestamp,
        "date_str": timestamp.strftime("%Y-%m-%d"),
    }
    result = _get_db().entries.insert_one(doc)
    return str(result.inserted_id)


def get_entries_by_date(date_str) -> list[dict]:
    docs = _get_db().entries.find(
        {"user_id": DEFAULT_USER_ID, "date_str": date_str}
    ).sort("timestamp", 1)
    return _serialize_docs(docs)


def get_entries_by_range(start_date, end_date) -> list[dict]:
    docs = _get_db().entries.find(
        {
            "user_id": DEFAULT_USER_ID,
            "date_str": {"$gte": start_date, "$lte": end_date},
        }
    ).sort("timestamp", 1)
    return _serialize_docs(docs)


def save_structured_entries(entries: list[dict]) -> None:
    if not entries:
        return

    docs = []
    for entry in entries:
        doc = dict(entry)
        timestamp = doc.get("timestamp") or _utc_now()
        doc["timestamp"] = timestamp
        doc.setdefault("user_id", DEFAULT_USER_ID)
        doc.setdefault("date_str", timestamp.strftime("%Y-%m-%d"))
        docs.append(doc)

    _get_db().structured_entries.insert_many(docs)


def get_structured_entries_by_date(date_str) -> list[dict]:
    docs = _get_db().structured_entries.find(
        {"user_id": DEFAULT_USER_ID, "date_str": date_str}
    ).sort("timestamp", 1)
    return _serialize_docs(docs)


def get_structured_entries_by_range(start_date, end_date) -> list[dict]:
    docs = _get_db().structured_entries.find(
        {
            "user_id": DEFAULT_USER_ID,
            "date_str": {"$gte": start_date, "$lte": end_date},
        }
    ).sort("timestamp", 1)
    return _serialize_docs(docs)


def save_report(report_type, period, pdf_binary, summary) -> None:
    _get_db().reports.update_one(
        {"user_id": DEFAULT_USER_ID, "report_type": report_type, "period": period},
        {
            "$set": {
                "user_id": DEFAULT_USER_ID,
                "report_type": report_type,
                "period": period,
                "pdf_binary": pdf_binary,
                "summary": summary,
                "updated_at": _utc_now(),
            },
            "$setOnInsert": {"created_at": _utc_now()},
        },
        upsert=True,
    )


def get_report(report_type, period) -> dict | None:
    doc = _get_db().reports.find_one(
        {"user_id": DEFAULT_USER_ID, "report_type": report_type, "period": period}
    )
    return _serialize_doc(doc) if doc else None


def get_all_entries_for_chat(limit=200) -> list[dict]:
    docs = (
        _get_db()
        .structured_entries.find({"user_id": DEFAULT_USER_ID})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(reversed(_serialize_docs(docs)))


def today_str() -> str:
    return _utc_now().strftime("%Y-%m-%d")


def current_month_str() -> str:
    return _utc_now().strftime("%Y-%m")


def _entry_time(entry: dict[str, Any]) -> str:
    timestamp = entry.get("timestamp")
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%H:%M")
    return ""


def _safe_class_value(value: Any) -> str:
    return html.escape(str(value or "unknown").strip().lower().replace(" ", "-"))


def _safe_text(value: Any) -> str:
    return html.escape(str(value or ""))


def render_entries_html(date_str) -> str:
    structured_entries = get_structured_entries_by_date(date_str)

    if structured_entries:
        cards = []
        for entry in structured_entries:
            category = entry.get("category", "unknown")
            severity = entry.get("severity", "unknown")
            structured_text = entry.get("structured_text") or entry.get("text") or ""
            cards.append(
                '<div class="entry-card">'
                f'<div class="entry-time">{_entry_time(entry)}</div>'
                f'<div class="entry-category-badge category-{_safe_class_value(category)}">'
                f"{_safe_text(category)}</div>"
                f'<div class="entry-text">{_safe_text(structured_text)}</div>'
                f'<div class="entry-severity severity-{_safe_class_value(severity)}">'
                f"{_safe_text(severity)}</div>"
                "</div>"
            )
        return "".join(cards)

    raw_entries = get_entries_by_date(date_str)
    if raw_entries:
        cards = []
        for entry in raw_entries:
            cards.append(
                '<div class="entry-card raw">'
                f'<div class="entry-time">{_entry_time(entry)}</div>'
                f'<div class="entry-text">{_safe_text(entry.get("raw_text"))}</div>'
                "</div>"
            )
        return "".join(cards)

    return (
        '<p class="no-entries">'
        "No entries yet today. Start logging how you feel."
        "</p>"
    )

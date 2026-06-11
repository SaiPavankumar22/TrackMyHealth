import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any

import modal


MODEL_DIR = "/models"
MODEL_PATH = f"{MODEL_DIR}/nemotron-nano-4b.gguf"

app = modal.App("health-tracker")

image = (
    modal.Image.debian_slim()
    .pip_install("llama-cpp-python", "pymongo", "reportlab", "huggingface-hub")
    .run_commands(
        "apt-get update && apt-get install -y libglib2.0-0 libpango-1.0-0 "
        "libpangocairo-1.0-0 libcairo2"
    )
)

model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)


def _timestamp_value(entry: dict[str, Any]) -> datetime | str | None:
    return entry.get("timestamp") or entry.get("created_at")


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _time_label(value: Any) -> str:
    parsed = _parse_datetime(value)
    if parsed:
        return parsed.strftime("%H:%M")
    if isinstance(value, str):
        match = re.search(r"(\d{1,2}:\d{2})", value)
        if match:
            return match.group(1).zfill(5)
    return "00:00"


def _date_label(entry: dict[str, Any]) -> str:
    if entry.get("date_str"):
        return str(entry["date_str"])
    parsed = _parse_datetime(_timestamp_value(entry))
    return parsed.strftime("%Y-%m-%d") if parsed else ""


def _timestamp_label(entry: dict[str, Any]) -> str:
    value = _timestamp_value(entry)
    parsed = _parse_datetime(value)
    if parsed:
        return parsed.isoformat()
    return str(value or "")


def _entry_id(entry: dict[str, Any]) -> str:
    return str(entry.get("entry_id") or entry.get("_id") or entry.get("id") or "")


def _fallback_structured_entry(entry: dict[str, Any]) -> dict[str, Any]:
    timestamp = _timestamp_value(entry)
    raw_text = str(entry.get("raw_text") or entry.get("text") or "")
    return {
        "entry_id": _entry_id(entry),
        "user_id": entry.get("user_id"),
        "timestamp": timestamp,
        "date_str": _date_label(entry),
        "structured_text": raw_text,
        "category": "general",
        "severity": "mild",
        "keywords": _keywords(raw_text),
    }


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text.lower())
    stopwords = {
        "and",
        "the",
        "with",
        "for",
        "that",
        "this",
        "was",
        "were",
        "but",
        "had",
        "have",
        "has",
        "felt",
        "feel",
    }
    terms = [word for word in words if word not in stopwords]
    return list(dict.fromkeys(terms))[:5] or ["general", "health"]


def _extract_json_array(text: str) -> list[Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("LLM response was not a JSON array")
    return parsed


def _llm_text(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"]).strip()


@app.function(image=image, timeout=600, volumes={MODEL_DIR: model_cache})
def download_model() -> str:
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "nvidia/Nemotron-Nano-4B-Instruct"
    filename = os.environ.get("NEMOTRON_GGUF_FILENAME")
    if not filename:
        files = HfApi().list_repo_files(repo_id)
        gguf_files = [path for path in files if path.lower().endswith(".gguf")]
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files found in {repo_id}")
        filename = sorted(gguf_files)[0]

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    if downloaded != MODEL_PATH:
        os.replace(downloaded, MODEL_PATH)
    model_cache.commit()
    return MODEL_PATH


def load_model():
    from llama_cpp import Llama

    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )


@app.function(image=image, gpu="T4", timeout=120, volumes={MODEL_DIR: model_cache})
def structure_entries(raw_entries: list[dict]) -> list[dict]:
    if not raw_entries:
        return []

    llm = load_model()
    formatted = "\n".join(
        f"{index}. [{_timestamp_label(entry)}] "
        f"{entry.get('raw_text') or entry.get('text') or ''}"
        for index, entry in enumerate(raw_entries, 1)
    )
    system_prompt = (
        "You are a medical scribe. Convert raw health journal entries into concise "
        "clinical observations. For each entry return a JSON object with: "
        "structured_text (one clear sentence starting with HH:MM \u2014), category "
        "(one of: sleep/pain/mood/digestion/energy/medication/general), severity "
        "(one of: mild/moderate/notable), keywords (array of 2-5 terms). Respond "
        "with JSON array only, no markdown."
    )
    user_prompt = (
        f"Process these {len(raw_entries)} entries:\n{formatted}\n"
        f"Return JSON array of {len(raw_entries)} objects in same order."
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    try:
        parsed = _extract_json_array(_llm_text(response))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        parsed = []

    structured_entries = []
    for index, entry in enumerate(raw_entries):
        fallback = _fallback_structured_entry(entry)
        llm_entry = parsed[index] if index < len(parsed) and isinstance(parsed[index], dict) else {}
        structured_text = str(llm_entry.get("structured_text") or fallback["structured_text"])
        category = str(llm_entry.get("category") or "general").lower()
        severity = str(llm_entry.get("severity") or "mild").lower()
        keywords = llm_entry.get("keywords") if isinstance(llm_entry.get("keywords"), list) else []

        if category not in {"sleep", "pain", "mood", "digestion", "energy", "medication", "general"}:
            category = "general"
        if severity not in {"mild", "moderate", "notable"}:
            severity = "mild"

        structured_entries.append(
            {
                "entry_id": fallback["entry_id"],
                "user_id": fallback["user_id"],
                "timestamp": fallback["timestamp"],
                "date_str": fallback["date_str"],
                "structured_text": structured_text,
                "category": category,
                "severity": severity,
                "keywords": [str(keyword) for keyword in keywords[:5]] or fallback["keywords"],
            }
        )

    return structured_entries


def _category_color(category: str):
    from reportlab.lib import colors

    hex_colors = {
        "sleep": "#4A90D9",
        "pain": "#E05252",
        "mood": "#F5A623",
        "digestion": "#7ED321",
        "energy": "#9B59B6",
        "medication": "#1ABC9C",
        "general": "#95A5A6",
    }
    return colors.HexColor(hex_colors.get(category, hex_colors["general"]))


def _draw_wrapped_text(canvas, text: str, x: float, y: float, max_width: float, leading: int = 12):
    from reportlab.pdfbase.pdfmetrics import stringWidth

    words = str(text).split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if stringWidth(candidate, "Helvetica", 9) <= max_width:
            line = candidate
        else:
            canvas.drawString(x, y, line)
            y -= leading
            line = word
    if line:
        canvas.drawString(x, y, line)
        y -= leading
    return y


def _summary_from_llm(entries: list[dict], date_str: str) -> str:
    llm = load_model()
    entry_text = "\n".join(
        f"- [{_time_label(entry.get('timestamp'))}] "
        f"({entry.get('category', 'general')}, {entry.get('severity', 'mild')}) "
        f"{entry.get('structured_text', '')}"
        for entry in entries
    )
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a personal health analyst. Be concise and compassionate. "
                    "Never diagnose. Recommend a doctor for serious issues."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create a daily health report for {date_str} using these entries:\n"
                    f"{entry_text}\n\nInclude: Overall Status (2-3 sentences), Key "
                    "Observations by category, Patterns or Concerns, Suggestions."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    return _llm_text(response)


@app.function(image=image, gpu="T4", timeout=180, volumes={MODEL_DIR: model_cache})
def generate_daily_pdf(date_str: str, structured_entries: list[dict]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    summary = _summary_from_llm(structured_entries, date_str)
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 20)
    c.drawString(48, height - 54, f"Daily Health Report \u2014 {date_str}")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.gray)
    c.drawString(48, height - 72, f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    c.setFillColor(colors.black)
    c.line(48, height - 84, width - 48, height - 84)
    c.setFont("Helvetica", 11)
    y = _draw_wrapped_text(c, summary, 48, height - 110, width - 96, 14)
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, y, "Key Observations")
    y -= 18
    c.setFont("Helvetica", 10)
    for entry in structured_entries[:18]:
        y = _draw_wrapped_text(
            c,
            f"- {entry.get('category', 'general')}: {entry.get('structured_text', '')}",
            58,
            y,
            width - 116,
            12,
        )
        y -= 2
        if y < 64:
            break
    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(48, height - 54, "Timeline")
    top = height - 90
    bottom = 92
    c.setStrokeColor(colors.lightgrey)
    c.line(150, top, 150, bottom)
    row_height = 40
    y = top
    for entry in structured_entries:
        if y < bottom + row_height:
            c.showPage()
            c.setFont("Helvetica-Bold", 16)
            c.drawString(48, height - 54, "Timeline")
            top = height - 90
            y = top
            c.setStrokeColor(colors.lightgrey)
            c.line(150, top, 150, bottom)
        category = str(entry.get("category", "general"))
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.black)
        c.drawRightString(132, y - 5, _time_label(entry.get("timestamp")))
        c.setFillColor(_category_color(category))
        c.circle(150, y - 2, 5, fill=1, stroke=0)
        c.setFillColor(colors.black)
        _draw_wrapped_text(c, entry.get("structured_text", ""), 172, y + 4, width - 220, 10)
        y -= row_height

    c.setFont("Helvetica", 8)
    legend_x = 48
    for category in ["sleep", "pain", "mood", "digestion", "energy", "medication", "general"]:
        c.setFillColor(_category_color(category))
        c.circle(legend_x, 42, 4, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(legend_x + 8, 39, category)
        legend_x += 72
    c.save()
    return buffer.getvalue()


def _period_summary(entries: list[dict], report_type: str, start_date: str, end_date: str) -> tuple[str, str]:
    llm = load_model()
    by_date = defaultdict(list)
    for entry in entries:
        by_date[_date_label(entry)].append(entry)
    grouped_text = "\n\n".join(
        f"{date}:\n"
        + "\n".join(
            f"- [{_time_label(entry.get('timestamp'))}] ({entry.get('category')}) "
            f"{entry.get('structured_text', '')}"
            for entry in day_entries
        )
        for date, day_entries in sorted(by_date.items())
    )
    day_response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Summarize health journal days concisely. Never diagnose."},
            {
                "role": "user",
                "content": (
                    "For each date below, write one short paragraph summarizing the day.\n\n"
                    f"{grouped_text}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1600,
    )
    day_summaries = _llm_text(day_response)
    trend_response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a personal health analyst. Be concise and compassionate. "
                    "Never diagnose. Recommend a doctor for serious issues."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create a {report_type} health trend report for {start_date} to {end_date}.\n"
                    f"Day summaries:\n{day_summaries}\n\nInclude overall trend, frequent symptoms "
                    "with counts, notable days, pattern analysis, and suggestions."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1600,
    )
    return day_summaries, _llm_text(trend_response)


@app.function(image=image, gpu="T4", timeout=300, volumes={MODEL_DIR: model_cache})
def generate_range_pdf(
    report_type: str, start_date: str, end_date: str, structured_entries: list[dict]
) -> bytes:
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    day_summaries, trend_summary = _period_summary(
        structured_entries, report_type, start_date, end_date
    )
    by_date = defaultdict(list)
    for entry in structured_entries:
        by_date[_date_label(entry)].append(entry)
    category_counts = Counter(str(entry.get("category", "general")) for entry in structured_entries)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=48, leftMargin=48)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"{report_type.title()} Health Report", styles["Title"]),
        Paragraph(f"{start_date} to {end_date}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph(trend_summary.replace("\n", "<br/>"), styles["BodyText"]),
        Spacer(1, 16),
        Paragraph("Frequency Table", styles["Heading2"]),
    ]
    table_data = [["Category", "Count"]] + [[category, count] for category, count in category_counts.items()]
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([table, PageBreak(), Paragraph("Day-by-Day", styles["Title"])])
    for date, entries in sorted(by_date.items()):
        story.append(Paragraph(date, styles["Heading2"]))
        for entry in entries:
            story.append(
                Paragraph(
                    f"{_time_label(entry.get('timestamp'))} - "
                    f"{entry.get('category', 'general')}: {entry.get('structured_text', '')}",
                    styles["BodyText"],
                )
            )
        story.append(Spacer(1, 10))
    story.extend([PageBreak(), Paragraph("Entries Per Day", styles["Title"])])
    dates = sorted(by_date)
    chart = VerticalBarChart()
    chart.x = 40
    chart.y = 40
    chart.height = 220
    chart.width = 420
    chart.data = [[len(by_date[date]) for date in dates]]
    chart.categoryAxis.categoryNames = dates
    chart.categoryAxis.labels.angle = 45
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = colors.HexColor("#4A90D9")
    drawing = Drawing(500, 300)
    drawing.add(chart)
    story.append(drawing)
    story.extend([PageBreak(), Paragraph("Daily Summaries", styles["Title"])])
    story.append(Paragraph(day_summaries.replace("\n", "<br/>"), styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()


@app.function(image=image, gpu="T4", timeout=60, volumes={MODEL_DIR: model_cache})
def chat_with_history(
    user_message: str, recent_entries: list[dict], chat_history: list[dict]
) -> str:
    llm = load_model()
    formatted_entries = "\n".join(
        f"[{_date_label(entry)} {_time_label(entry.get('timestamp'))}] "
        f"({entry.get('category', 'general')}, {entry.get('severity', 'mild')}) "
        f"{entry.get('structured_text', '')}"
        for entry in recent_entries
    )
    system_prompt = (
        "You are a personal health assistant with access to the user's health journal. "
        "Ground all answers in their actual data. Reference specific dates when relevant. "
        "Never diagnose. Recommend a doctor for serious issues. If data is missing, say so. "
        f"Be warm and concise.\n\nUser's journal:\n{formatted_entries}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-10:]:
        role = turn.get("role")
        content = turn.get("content") or turn.get("message")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_message})

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )
    return _llm_text(response)

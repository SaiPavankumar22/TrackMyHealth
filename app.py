import calendar
import os
from datetime import datetime, timedelta

import gradio as gr
import modal

import db
import modal_app


TMP_DIR = "/tmp"


def _tmp_pdf_path(filename: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return os.path.join(TMP_DIR, filename)


def _write_pdf(path: str, pdf: bytes) -> str:
    with open(path, "wb") as file:
        file.write(pdf)
    return path


def log_entry(text):
    if not text or not text.strip():
        return "", " Enter something first.", gr.update()

    now = datetime.utcnow()
    db.save_raw_entry(text.strip(), now)
    today = db.today_str()
    return "", f" Logged at {now.strftime('%H:%M')} UTC", db.render_entries_html(today)


def refresh_entries():
    return db.render_entries_html(db.today_str())


def generate_daily():
    today = db.today_str()
    yield " Fetching entries...", gr.update(visible=False)

    raw = db.get_entries_by_date(today)
    if not raw:
        yield " No entries found for today.", gr.update(visible=False)
        return

    yield "🧠 Structuring with Nemotron Nano...", gr.update(visible=False)
    structured = modal_app.structure_entries.remote(raw)
    db.save_structured_entries(structured)

    yield "📄 Generating PDF...", gr.update(visible=False)
    pdf = modal_app.generate_daily_pdf.remote(today, structured)
    path = _tmp_pdf_path(f"health_report_{today}.pdf")
    _write_pdf(path, pdf)
    yield " Report ready.", gr.update(visible=True, value=path)


def chat_respond(message, history):
    if not message or not message.strip():
        return history, ""

    history = history or []
    entries = db.get_all_entries_for_chat(200)
    response = modal_app.chat_with_history.remote(message.strip(), entries, history)
    history.append({"role": "user", "content": message.strip()})
    history.append({"role": "assistant", "content": response})
    return history, ""


def clear_chat():
    return []


def generate_weekly(week_end_str):
    yield " Fetching entries...", gr.update(visible=False)
    try:
        week_end = datetime.strptime(week_end_str.strip(), "%Y-%m-%d").date()
    except (AttributeError, ValueError):
        yield " Enter week ending date as YYYY-MM-DD.", gr.update(visible=False)
        return

    week_start = week_end - timedelta(days=6)
    start = week_start.strftime("%Y-%m-%d")
    end = week_end.strftime("%Y-%m-%d")
    raw = db.get_entries_by_range(start, end)
    if not raw:
        yield " No entries found for that week.", gr.update(visible=False)
        return

    yield "🧠 Structuring with Nemotron Nano...", gr.update(visible=False)
    structured = modal_app.structure_entries.remote(raw)
    db.save_structured_entries(structured)

    yield "📄 Generating PDF...", gr.update(visible=False)
    pdf = modal_app.generate_range_pdf.remote("weekly", start, end, structured)
    path = _tmp_pdf_path(f"health_report_weekly_{start}_to_{end}.pdf")
    _write_pdf(path, pdf)
    yield " Report ready.", gr.update(visible=True, value=path)


def generate_monthly(month_str):
    yield " Fetching entries...", gr.update(visible=False)
    try:
        month = datetime.strptime(month_str.strip(), "%Y-%m")
    except (AttributeError, ValueError):
        yield " Enter month as YYYY-MM.", gr.update(visible=False)
        return

    last_day = calendar.monthrange(month.year, month.month)[1]
    start = month.replace(day=1).strftime("%Y-%m-%d")
    end = month.replace(day=last_day).strftime("%Y-%m-%d")
    raw = db.get_entries_by_range(start, end)
    if not raw:
        yield " No entries found for that month.", gr.update(visible=False)
        return

    yield "🧠 Structuring with Nemotron Nano...", gr.update(visible=False)
    structured = modal_app.structure_entries.remote(raw)
    db.save_structured_entries(structured)

    yield "📄 Generating PDF...", gr.update(visible=False)
    pdf = modal_app.generate_range_pdf.remote("monthly", start, end, structured)
    path = _tmp_pdf_path(f"health_report_monthly_{month.strftime('%Y-%m')}.pdf")
    _write_pdf(path, pdf)
    yield " Report ready.", gr.update(visible=True, value=path)


with gr.Blocks(
    theme=gr.themes.Base(),
    css=open("static/style.css", encoding="utf-8").read(),
    title="HealthLog",
) as demo:
    gr.Markdown("# 🩺 HealthLog")
    gr.Markdown("Your private daily health journal.")

    with gr.Tab("Today"):
        with gr.Group():
            gr.Markdown("### How are you feeling?")
            entry_input = gr.Textbox(
                lines=3,
                placeholder="Describe symptoms, sleep, pain, mood... anything.",
                elem_classes="entry-input",
            )
            log_btn = gr.Button("Log Entry", variant="primary", elem_classes="log-btn")
            log_status = gr.Markdown()

        gr.Markdown("### Today's Log", elem_classes="section-header")
        entries_display = gr.HTML(value=db.render_entries_html(db.today_str()))
        refresh_btn = gr.Button(" Refresh", size="sm")
        daily_report_btn = gr.Button("Generate Daily Report PDF", elem_classes="report-btn")
        daily_status = gr.Markdown("")
        daily_download = gr.File(visible=False)

    with gr.Tab("Chat"):
        gr.Markdown("Ask anything about your health history.")
        chatbot = gr.Chatbot(
            height=460,
            elem_id="health-chatbot",
            bubble_full_width=False,
            show_label=False,
            type="messages",
        )
        chat_input = gr.Textbox(
            placeholder="Have my headaches been getting worse this week?",
            show_label=False,
            elem_classes="chat-input",
        )
        with gr.Row():
            chat_submit = gr.Button("Send", variant="primary")
            chat_clear = gr.Button("Clear", size="sm")

    with gr.Tab("Reports"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Weekly Report")
                week_end = gr.Textbox(label="Week ending (YYYY-MM-DD)", value=db.today_str())
                weekly_btn = gr.Button("Generate Weekly Report", elem_classes="report-btn")
                weekly_status = gr.Markdown("")
                weekly_download = gr.File(visible=False)

            with gr.Column():
                gr.Markdown("### Monthly Report")
                month_input = gr.Textbox(label="Month (YYYY-MM)", value=db.current_month_str())
                monthly_btn = gr.Button("Generate Monthly Report", elem_classes="report-btn")
                monthly_status = gr.Markdown("")
                monthly_download = gr.File(visible=False)

    log_btn.click(log_entry, inputs=entry_input, outputs=[entry_input, log_status, entries_display])
    entry_input.submit(
        log_entry,
        inputs=entry_input,
        outputs=[entry_input, log_status, entries_display],
    )
    refresh_btn.click(refresh_entries, outputs=entries_display)
    daily_report_btn.click(generate_daily, outputs=[daily_status, daily_download])

    chat_submit.click(
        chat_respond,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(
        chat_respond,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    )
    chat_clear.click(clear_chat, outputs=chatbot)

    weekly_btn.click(generate_weekly, inputs=week_end, outputs=[weekly_status, weekly_download])
    monthly_btn.click(generate_monthly, inputs=month_input, outputs=[monthly_status, monthly_download])


if __name__ == "__main__":
    demo.launch()

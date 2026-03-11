import os
import re
from pathlib import Path

import gradio as gr

# ---------------- OpenAI Client (for AI Sherlock) ---------------- #

try:
    from openai import OpenAI

    client = OpenAI()
except Exception:
    client = None


# ---------------- Sherlock Logic ---------------- #


def extract_clues(text: str):
    """
    Very simple heuristic clue extractor.
    """
    clues = {
        "suspects": set(),
        "times": set(),
        "claims": [],
    }

    # 1) Look for lines like "Suspect: John Doe"
    for line in text.splitlines():
        if "suspect:" in line.lower():
            try:
                part = line.split(":", 1)[1].strip()
                if part:
                    clues["suspects"].add(part)
            except Exception:
                continue

    # 2) Look for times like "8 PM", "7AM", "10:30 pm"
    time_pattern = r"\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)\b"
    for m in re.finditer(time_pattern, text):
        clues["times"].add(m.group(0))

    # 3) Naive "claims" – sentences with keywords like 'claims', 'said', 'reported'
    claim_keywords = ["claims", "claimed", "says", "said", "reports", "reported"]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        lower = sent.lower()
        if any(kw in lower for kw in claim_keywords):
            sent = sent.strip()
            if sent:
                clues["claims"].append(sent)

    clues["suspects"] = sorted(clues["suspects"])
    clues["times"] = sorted(clues["times"])
    return clues


def assess_risk(clues: dict, text: str):
    """
    Simple heuristic risk assessment based on clues + keywords.
    Returns (risk_level, reasons_list).
    """
    reasons = []
    lower_text = text.lower()

    financial_keywords = [
        "bank",
        "payment",
        "transfer",
        "otp",
        "password",
        "account",
        "fraud",
        "scam",
        "urgent",
    ]
    contradiction_markers = ["however", "but", "whereas", "although", "yet"]

    has_financial_terms = any(k in lower_text for k in financial_keywords)
    has_contradiction_words = any(m in lower_text for m in contradiction_markers)
    many_claims = len(clues.get("claims", [])) >= 3
    many_times = len(clues.get("times", [])) >= 2

    if has_financial_terms:
        reasons.append("Mentions of financial / account-related terms.")
    if has_contradiction_words:
        reasons.append("Contains potential contradiction markers (e.g. 'however', 'but').")
    if many_claims:
        reasons.append("Multiple claims detected that may need verification.")
    if many_times:
        reasons.append("Multiple time references detected; timeline may be important.")

    if has_financial_terms and has_contradiction_words:
        risk = "High"
    elif has_financial_terms or has_contradiction_words or many_claims:
        risk = "Medium"
    else:
        risk = "Low"

    if not reasons:
        reasons.append("No obvious red flags detected by heuristic rules.")

    return risk, reasons


def build_timeline(text: str):
    """
    Extract a simple timeline of events from the evidence text.

    Heuristic: for each sentence with a time expression,
    create a bullet point "<time> — <sentence>".
    """
    time_pattern = r"\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)\b"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    timeline_items = []

    for sent in sentences:
        matches = list(re.finditer(time_pattern, sent))
        if matches:
            times = ", ".join(m.group(0) for m in matches)
            clean_sent = sent.strip()
            if clean_sent:
                timeline_items.append(f"- **{times}** — {clean_sent}")

    if not timeline_items:
        return "_No explicit time-based events detected by the heuristic._"

    return "\n".join(timeline_items)


def generate_ai_summary(text: str, clues: dict) -> str:
    """
    LLM-powered Sherlock-style summary using OpenAI.
    Safe to call even if no API key is set: returns a fallback message.
    """
    if client is None or not os.getenv("OPENAI_API_KEY"):
        return "_AI Sherlock Summary is disabled (no OPENAI_API_KEY configured)._"

    suspects = ", ".join(clues.get("suspects") or []) or "(none)"
    times = ", ".join(clues.get("times") or []) or "(none)"
    claims = clues.get("claims") or []
    clue_text = f"Suspects: {suspects}\nTimes: {times}\nNumber of claims: {len(claims)}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are Sherlock Holmes writing a concise investigative summary. "
                "You receive raw evidence text plus some pre-extracted clues. "
                "Write 1–2 short paragraphs in clear, modern English, explaining "
                "what seems to be happening, what looks suspicious, and what a "
                "human investigator should check next. Be concrete and avoid fluff."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Here are the pre-extracted clues:\n{clue_text}\n\n"
                f"Here is the full evidence text:\n{text[:4000]}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"_AI Sherlock Summary unavailable (error: {e})._"


# ---------------- LLM Tools: Contradictions & Phishing ---------------- #


def detect_contradictions_core(text: str, clues=None, timeline_md=None) -> str:
    """
    LLM-based contradiction analysis.
    """
    if client is None or not os.getenv("OPENAI_API_KEY"):
        return "_Contradiction detector is disabled (no OPENAI_API_KEY configured)._"

    if clues is None:
        clues = extract_clues(text)
    if timeline_md is None:
        timeline_md = build_timeline(text)

    claims = clues.get("claims") or []
    claims_text = "\n".join(f"- {c}" for c in claims) if claims else "(no explicit claims detected)"

    prompt = (
        "You are an analytical investigator. Your job is to find contradictions,\n"
        "inconsistencies, or suspicious gaps in the following case.\n\n"
        "You are given:\n\n"
        "CLAIMS:\n"
        f"{claims_text}\n\n"
        "TIMELINE:\n"
        f"{timeline_md}\n\n"
        "FULL EVIDENCE TEXT:\n"
        f"{text[:4000]}\n\n"
        "Respond with a short markdown section with:\n"
        "1. A bullet list of any contradictions you find.\n"
        "2. Notes on any missing information or ambiguous points.\n"
        "3. A 1–2 sentence conclusion about how consistent the story is.\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"_Contradiction detector error: {e}_"


def assess_phishing_core(text: str) -> str:
    """
    LLM-based phishing / scam risk assessment.
    Returns a markdown string summarizing score + reasoning.
    """
    if client is None or not os.getenv("OPENAI_API_KEY"):
        return "_Phishing assessor is disabled (no OPENAI_API_KEY configured)._"

    prompt = (
        "You are a fraud and phishing expert.\n"
        "Given the following message, rate how likely it is to be phishing or a scam.\n\n"
        "Return a short markdown answer with:\n"
        "- A score from 0–100 (0 = definitely safe, 100 = definitely phishing).\n"
        "- 3–5 concrete reasons referencing the text.\n"
        "- A short recommendation for the user (e.g. ignore, verify with bank, etc.).\n\n"
        f"Message:\n{text[:4000]}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"_Phishing assessor error: {e}_"


# ---------------- MCP Tool Wrappers ---------------- #


def detect_contradictions_tool(text: str):
    if not text or not text.strip():
        return "No text provided. Please pass some evidence text."

    clues = extract_clues(text)
    timeline_md = build_timeline(text)
    analysis = detect_contradictions_core(text, clues, timeline_md)
    return f"### ❗ Contradiction Analysis\n\n{analysis}"


def assess_phishing_tool(text: str):
    if not text or not text.strip():
        return "No text provided. Please pass some evidence text."

    analysis = assess_phishing_core(text)
    return f"### 🎣 Phishing / Scam Risk\n\n{analysis}"


# ---------------- Sample Case ---------------- #


SAMPLE_CASE_TEXT = """From: "PayPal Security" <security-update@paypalsecure-info.com>
Subject: URGENT: Account Access Suspended

Dear Customer,
We detected unfamiliar activity on your account at 2:14 AM.
Your account has been temporarily locked.

To unlock, please verify your identity at the link below:
https://paypal-verify-login-reset.com/id/secure

If you do not verify in the next 2 hours, your account will be permanently suspended.

Do NOT reply to this email.
"""


# ---------------- Agent Orchestrator ---------------- #


def sherlock_agent_orchestrator(text: str, mode: str):
    """
    Mini agent that plans and executes the investigation steps.
    """
    base_plan = [
        "Ingest the evidence text into the workspace.",
        "Extract suspects, time references, and claims.",
        "Assess risk level based on keywords and contradictions.",
        "Build a timeline of events from time expressions.",
        "Generate an AI Sherlock-style narrative summary.",
    ]

    if mode == "Forensics Mode":
        extra = ["Optionally run contradiction analysis on the claims and timeline."]
    elif mode == "Scam Email Mode":
        extra = ["Assess phishing / scam risk for the message."]
    elif mode == "Legal / Compliance Mode":
        extra = ["Highlight claims relevant to responsibility, liability or policy breaches."]
    else:
        extra = []

    plan_steps = base_plan + extra

    log_lines = []
    log_lines.append(f"### 🧠 Sherlock Agent Plan ({mode})")
    for i, step in enumerate(plan_steps, start=1):
        log_lines.append(f"{i}. {step}")
    log_lines.append("")
    log_lines.append("### 🚀 Executing Plan")
    log_lines.append("Step 1: Ingest evidence ✅")

    # Step 2: extract clues
    clues = extract_clues(text)
    log_lines.append(
        f"Step 2: Extract clues ✅ "
        f"(suspects: {len(clues['suspects'])}, times: {len(clues['times'])}, claims: {len(clues['claims'])})"
    )

    # Step 3: risk assessment
    risk_level, reasons = assess_risk(clues, text)
    log_lines.append(f"Step 3: Assess risk ✅ (risk level: {risk_level})")

    # Step 4: timeline
    timeline_md = build_timeline(text)
    log_lines.append("Step 4: Build timeline ✅")

    # Step 5: AI summary
    ai_summary = generate_ai_summary(text, clues)
    log_lines.append("Step 5: Generate AI Sherlock summary ✅")

    # Optional extras
    if mode == "Scam Email Mode":
        log_lines.append("Step 6 (mode-specific): Phishing / scam risk assessed ✅")
    elif mode == "Forensics Mode":
        log_lines.append("Step 6 (mode-specific): Contradiction analysis available via MCP tool ✅")
    elif mode == "Legal / Compliance Mode":
        log_lines.append("Step 6 (mode-specific): Legal / compliance-relevant claims highlighted ✅")

    agent_log_md = "\n".join(log_lines)
    return agent_log_md, clues, risk_level, reasons, timeline_md, ai_summary


# ---------------- MCP Tool 1: Full Investigation ---------------- #


def start_investigation(file=None, manual_text=None, mode="Standard Sherlock"):
    """
    Sherlock.MCP tool: analyze uploaded or pasted evidence.

    - If manual_text is provided, it is used and file is ignored.
    - If file is provided, it is read as text.
    - If neither is provided, returns an error message.
    """
    source = None
    text = ""
    filename = ""

    # Make sure mode has a default
    if not mode:
        mode = "Standard Sherlock"

    # Prefer manual text if present
    if manual_text and str(manual_text).strip():
        source = "pasted text"
        text = manual_text
        filename = "pasted_evidence.txt"

    elif file is not None:
        source = "uploaded file"
        try:
            # from Gradio UI: file is an object with .name
            # from MCP / curl (future): could be a path string
            if isinstance(file, str):
                file_path = Path(file)
            else:
                file_path = Path(file.name)
            filename = file_path.name
        except Exception:
            log = (
                "⚠️ Sherlock could not access the file path.\n\n"
                "Try again with a different file or rename the file."
            )
            report = "No case report generated."
            return log, report, ""

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log = (
                f"🕵️ Sherlock received the case file: **{filename}**, "
                f"but couldn't read it as text.\n\nError: `{e}`"
            )
            report = "Try a plain text (.txt / .md) document for now."
            return log, report, ""
    else:
        log = "🕵️ No evidence provided yet, Watson."
        report = "Upload a case file or paste text to begin the investigation."
        return log, report, ""

    if not text.strip():
        log = (
            f"🕵️ Sherlock received the evidence (**{filename or source}**), "
            "but it appears to be empty or non-text."
        )
        report = "Try again with a text-based document."
        return log, report, ""

    # Short evidence preview
    preview = text[:800]

    # Run through the agent orchestrator
    (
        agent_log_md,
        clues,
        risk_level,
        reasons,
        timeline_md,
        ai_summary,
    ) = sherlock_agent_orchestrator(text, mode)

    suspects = clues["suspects"] or ["(none detected)"]
    times = clues["times"] or ["(none detected)"]
    claims = clues["claims"] or ["(no explicit claims detected)"]

    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪️")

    # High-level investigation log
    log_lines = [
        f"🕵️ **Evidence source:** {source}",
        f"- Label / filename: **{filename or '(not provided)'}**",
        f"- Approx. length: ~{len(text)} characters",
        f"- Mode: **{mode}**",
        "",
        "📥 **Step 1 — Evidence ingestion**",
        "Evidence has been loaded into Sherlock's workspace.",
        "",
        "🔎 **Step 2 — Clue extraction (heuristic)**",
        f"- Suspects detected: **{len(clues['suspects'])}**",
        f"- Time mentions detected: **{len(clues['times'])}**",
        f"- Claims detected: **{len(claims)}**",
        "",
        "⚖️ **Step 3 — Risk assessment (heuristic)**",
        f"- Overall risk level: {risk_emoji} **{risk_level}**",
    ]
    for r in reasons:
        log_lines.append(f"  - {r}")

    log_lines += [
        "",
        "🕒 **Step 4 — Timeline construction (heuristic)**",
        "Sherlock builds a simple timeline of time-stamped events.",
        "",
        "🧠 **Step 5 — AI Sherlock narrative (OpenAI)**",
        "A short LLM-generated summary is added to the case report.",
        "",
        "Next steps for a future MCP-powered agent could include:",
        "- [ ] Cross-check entities and dates on the web",
        "- [ ] Run deeper contradiction checks using LLM tools",
        "- [ ] Use phishing / scam MCP tool for suspicious messages",
    ]

    # Combine agent execution log + high-level log
    log_md = agent_log_md + "\n\n---\n\n" + "\n".join(log_lines)

    # Case report
    claims_md = "\n".join([f"- {c}" for c in claims])
    reasons_md = "\n".join([f"- {r}" for r in reasons])

    report_md = (
        "### 🧾 Evidence Preview\n"
        "```text\n"
        f"{preview}\n"
        "```\n\n"
        "### 🔍 Extracted Clues (Heuristic)\n"
        f"- **Suspects:** {', '.join(suspects)}\n"
        f"- **Times mentioned:** {', '.join(times)}\n\n"
        "### ⚖️ Risk Assessment\n"
        f"- **Risk level:** {risk_emoji} **{risk_level}**\n"
        f"{reasons_md}\n\n"
        "### 💬 Claims Detected\n"
        f"{claims_md}\n\n"
        "### 🕒 Timeline of Events (Heuristic)\n"
        f"{timeline_md}\n\n"
        "### 🧠 AI Sherlock Summary (OpenAI)\n"
        f"{ai_summary}\n\n"
        "_Hybrid engine: agentic orchestration of heuristic tools + optional OpenAI narrative summary._"
    )

    # Combined text for export
    export_text = (
        "=== Sherlock.MCP Case Report ===\n\n"
        "## Investigation Log\n\n"
        f"{log_md}\n\n\n"
        "## Case Report\n\n"
        f"{report_md}\n"
    )

    return log_md, report_md, export_text


# ---------------- MCP Tool 2: Timeline-only ---------------- #


def timeline_tool(text: str):
    if not text or not text.strip():
        return "No text provided. Please pass some evidence text."

    timeline_md = build_timeline(text)
    return f"### 🕒 Timeline of Events\n\n{timeline_md}"


# ---------------- Export Case Report ---------------- #


def download_report(case_text: str):
    if not case_text or not case_text.strip():
        return None

    file_path = "sherlock_case_report.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(case_text)

    return file_path


# ---------------- Gradio UI / CSS ---------------- #

CSS = """
/* ====== GLOBAL LAYOUT ====== */

.sherlock-root {
  max-width: 1180px;
  margin-left: auto;
  margin-right: auto;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  padding: 18px 18px 32px;
}

/* Dark outer background – helps “newspaper” cards pop */
body,
.gradio-container {
  background: #0b1120 !important;   /* very dark navy */
}

/* Make all default text slightly darker for readability on parchment */
.sherlock-root,
.sherlock-root * {
  color: #111827;
}

/* ====== HEADER / TITLE BANNER ====== */

.sherlock-header {
  background: #fef3c7;              /* soft warm parchment header */
  padding: 20px 22px;
  border-radius: 18px;
  margin-bottom: 18px;
  border: 1px solid #facc15;        /* subtle golden border */
  box-shadow: 0 14px 35px rgba(15, 23, 42, 0.65);
}

.sherlock-header h1 {
  font-size: 2.1rem;
  margin-bottom: 4px;
  font-weight: 750;
  letter-spacing: 0.02em;
}

.sherlock-header p {
  font-size: 1.02rem;
  margin: 0;
}

/* Thin “rule” under the header text to feel like a newspaper masthead */
.sherlock-header::after {
  content: "";
  display: block;
  margin-top: 10px;
  height: 1px;
  width: 100%;
  background: linear-gradient(to right, #facc15, #78350f, #facc15);
  opacity: 0.7;
}

/* ====== CARD SECTIONS (Case Input / Log / Report / Tools) ====== */

.sherlock-card {
  background: #fdf1d6;              /* parchment */
  border-radius: 18px;
  padding: 18px 18px 22px;
  border: 1px solid #e4c79b;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.55);
}

/* Section titles inside cards */
.sherlock-card h3,
.sherlock-card h4,
.gr-markdown h3,
.gr-markdown h4 {
  font-weight: 650;
  margin-bottom: 10px;
  color: #111827;
  font-size: 1.1rem;
}

/* ====== TYPOGRAPHY / MARKDOWN BODY ====== */

.gr-markdown,
.gr-markdown p,
.gr-markdown li {
  font-size: 1.02rem;
  line-height: 1.55;
  color: #111827;
}

/* Slightly larger list bullets for the investigation log */
.gr-markdown ul {
  padding-left: 1.25rem;
}

/* Labels under inputs */
.sherlock-card label,
.sherlock-card .gr-label {
  font-size: 0.98rem;
  font-weight: 600;
  color: #111827;
}

/* ====== EVIDENCE PREVIEW / CODE BLOCKS ====== */

.sherlock-root pre,
.sherlock-root code,
.sherlock-root .gr-markdown pre,
.sherlock-root .gr-markdown code {
  background-color: #fef9e7 !important;
  color: #111827 !important;
  border-radius: 10px;
  border: 1px solid #e4c79b;
  padding: 10px 12px;
  font-size: 0.96rem;
  overflow-x: auto;
}

/* ====== TEXT INPUTS / TEXTAREAS / FILE UPLOADER ====== */

.sherlock-root textarea,
.sherlock-root input[type="text"],
.sherlock-root input[type="search"],
.sherlock-root .gradio-textbox textarea {
  background: #fffdf5 !important;
  border: 1px solid #e5d0a8 !important;
  color: #111827 !important;
  border-radius: 10px;
}

.sherlock-root textarea::placeholder,
.sherlock-root input::placeholder {
  color: #9ca3af !important;
}

/* File uploader on parchment card */
.sherlock-root .gradio-file,
.sherlock-root .gradio-file * {
  background-color: #fef9e7 !important;
  color: #4b5563 !important;
  border-radius: 14px !important;
}

/* ====== DROPDOWN ====== */

.sherlock-root .gradio-dropdown,
.sherlock-root .gradio-dropdown *,
.sherlock-root select,
.sherlock-root [role="combobox"] {
  background-color: #fffdf5 !important;
  color: #111827 !important;
  border: 1px solid #e5d0a8 !important;
  border-radius: 10px !important;
}

/* Options menu */
.sherlock-root .gradio-dropdown ul,
.sherlock-root .gradio-dropdown li {
  background-color: #fffef5 !important;
  color: #111827 !important;
}

/* ====== BUTTONS ====== */

button {
  font-size: 0.98rem !important;
  border-radius: 999px !important;      /* pill buttons */
  font-weight: 600 !important;
  padding: 8px 16px !important;
  border: none !important;
}

/* Primary (Start Investigation) – bold accent */
button.primary {
  background: #ea580c !important;
  color: #fff7ed !important;
}

/* Secondary buttons (tools & download) – soft newspaper pill */
button:not(.primary) {
  background: #fef9e7 !important;
  color: #111827 !important;
  border: 1px solid #e4c79b !important;
}

/* Hover micro-animation */
button:hover {
  transform: translateY(-1px) scale(1.01);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.25);
  transition: 0.15s ease;
}

/* ====== SMALL DETAILS ====== */

/* Give right column cards a tiny top margin so they line up nicely on small screens */
@media (max-width: 900px) {
  .sherlock-card {
    margin-top: 8px;
  }
}

/* Tighten top spacing of cards under the header */
.sherlock-root > .gr-row,
.sherlock-root > .gr-block {
  margin-top: 4px;
}
"""


with gr.Blocks(elem_classes="sherlock-root") as demo:
    gr.HTML(f"<style>{CSS}</style>")

    case_state = gr.State("")

    # Header
    with gr.Column(elem_classes="sherlock-header"):
        gr.Markdown(
            """
# 🕵️‍♂️ Sherlock.MCP — AI Detective Agent  
A Gradio + MCP evidence analysis tool.

Upload a document **or** paste text. Sherlock extracts clues, assesses risk,
builds a timeline, and adds an AI-generated narrative summary using OpenAI.
            """
        )

    with gr.Row():
        # Left: Inputs
        with gr.Column(scale=1, min_width=360, elem_classes="sherlock-card"):
            gr.Markdown("### 📂 Case Input")

            file_input = gr.File(
                label="Upload Case File (optional)",
                file_types=["text"],
            )

            manual_text = gr.Textbox(
                label="✍️ Or paste evidence text",
                placeholder="Paste a suspicious email, incident report, story, or chat log...",
                lines=10,
            )

            mode_dropdown = gr.Dropdown(
                choices=[
                    "Standard Sherlock",
                    "Forensics Mode",
                    "Scam Email Mode",
                    "Legal / Compliance Mode",
                ],
                value="Standard Sherlock",
                label="Investigation mode",
            )

            sample_btn = gr.Button("🎲 Load Sample Case")
            start_btn = gr.Button("🔍 Start Investigation", variant="primary")

        # Right: Outputs
        with gr.Column(scale=2, min_width=480):
            with gr.Row():
                with gr.Column(elem_classes="sherlock-card"):
                    gr.Markdown("### 🧭 Investigation Log")
                    log_output = gr.Markdown()

            gr.Markdown("")

            with gr.Row():
                with gr.Column(elem_classes="sherlock-card"):
                    gr.Markdown("### 📄 Case Report")
                    report_output = gr.Markdown()
                    download_btn = gr.DownloadButton("⬇️ Download Case Report")

            gr.Markdown("")

            with gr.Row():
                with gr.Column(elem_classes="sherlock-card"):
                    gr.Markdown("### 🧰 Extra MCP Tools")
                    tools_input = gr.Textbox(
                        label="🔧 Evidence text for MCP tools (timeline, contradictions, phishing)",
                        placeholder="Paste any text here to run the standalone MCP tools without the full Sherlock pipeline...",
                        lines=6,
                    )

                    with gr.Row():
                        timeline_btn = gr.Button("🕒 Build Timeline (MCP Tool)")
                        contradictions_btn = gr.Button("❗ Detect Contradictions (MCP Tool)")
                        phishing_btn = gr.Button("🎣 Assess Phishing Risk (MCP Tool)")

                    timeline_output = gr.Markdown(label="Timeline")
                    contradictions_output = gr.Markdown(label="Contradictions")
                    phishing_output = gr.Markdown(label="Phishing / Scam Risk")

    # Wiring
    sample_btn.click(
        fn=lambda: SAMPLE_CASE_TEXT,
        inputs=None,
        outputs=manual_text,
    )

    start_btn.click(
        start_investigation,
        inputs=[file_input, manual_text, mode_dropdown],
        outputs=[log_output, report_output, case_state],
        api_name="investigate_case",
    )

    download_btn.click(
        download_report,
        inputs=case_state,
        outputs=download_btn,
    )

    timeline_btn.click(
        timeline_tool,
        inputs=tools_input,
        outputs=timeline_output,
        api_name="build_timeline",
    )

    contradictions_btn.click(
        detect_contradictions_tool,
        inputs=tools_input,
        outputs=contradictions_output,
        api_name="detect_contradictions",
    )

    phishing_btn.click(
        assess_phishing_tool,
        inputs=tools_input,
        outputs=phishing_output,
        api_name="assess_phishing",
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)

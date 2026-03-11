Sherlock.MCP — AI Detective Agent

Hybrid MCP Server + Agentic Evidence Analyzer
Track 1 + Track 2 Submission

HF Space: https://huggingface.co/spaces/MCP-1st-Birthday/sherlock-mcp

Demo Video: https://youtu.be/wkOqOuyI0jw

Social Post (Required): https://x.com/sidequest981/status/1995261341914083405

Hackathon Tracks

This project is submitted under both tracks:

Track 1 — Building MCP

building-mcp-track-consumer
(Consumer-facing MCP server exposing multiple tools)

Track 2 — MCP in Action

mcp-in-action-track-consumer
(Full Gradio agentic application using MCP tools)

Overview

Sherlock.MCP is an AI detective agent that analyzes suspicious emails, narratives, reports, and multi-step stories.
It performs clue extraction, timeline building, phishing scoring, risk assessment, contradiction detection, and generates an AI narrative summary.

It is built with:

Gradio 6

MCP Server (mcp_server=True)

Multiple MCP Tools

Agentic planning + orchestration

OpenAI GPT-4-mini models (for narrative, contradiction, phishing)

This makes Sherlock both:

1. A standalone MCP server

Fully callable from Claude Desktop, Cursor, ChatGPT Desktop, or curl.

2. A complete agentic Gradio application

Detective UI + autonomous investigation pipeline.

Features
1. Clue Extraction (Heuristic)

Sherlock detects:

Suspects

Time expressions

Claims / statements

Contradiction markers

2. Timeline Builder (MCP Tool)

Builds a markdown timeline:

- 8 PM — The suspect said he was home.
- 8:04 PM — CCTV shows activity elsewhere.

3. Phishing / Scam Detection (LLM)

LLM-powered scam score with reasons and recommendations.

4. Heuristic Risk Assessment

Sherlock checks:

Urgency cues

Financial keywords

Multiple conflicting claims

Timeline clustering

Contradiction cues

5. AI Sherlock Summary (OpenAI GPT)

A narrative 1–2 paragraph detective-style summary.
This qualifies for the OpenAI "Best API Integration" prize.

6. Exposed MCP Tools

These tools are available via MCP:

Tool Name	What it Does
start_investigation	Full pipeline: clues, timeline, risk, AI summary
timeline_tool	Returns a timeline from any text
detect_contradictions_tool	LLM-based contradiction detection
assess_phishing_tool	LLM-based phishing score
download_report	Exports a text report
Using Sherlock via MCP (curl examples)
List all tools
curl -N \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":"1","method":"tools/list"}' \
  https://huggingface.co/spaces/MCP-1st-Birthday/sherlock-mcp/gradio_api/mcp/

Timeline Tool
curl -N \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tools/call",
    "params": {
      "name": "timeline_tool",
      "arguments": {
        "text": "The suspect said he was home at 8 PM, but CCTV shows 8:04 PM."
      }
    }
  }' \
  https://huggingface.co/spaces/MCP-1st-Birthday/sherlock-mcp/gradio_api/mcp/

Phishing Risk Tool
curl -N \
 -H "Content-Type: application/json" \
 -H "Accept: application/json, text/event-stream" \
 -d '{
  "jsonrpc": "2.0",
  "id": "3",
  "method": "tools/call",
  "params": {
    "name": "assess_phishing_tool",
    "arguments": {
      "text": "URGENT: Verify your bank account by entering your password now."
    }
  }
 }' \
 https://huggingface.co/spaces/MCP-1st-Birthday/sherlock-mcp/gradio_api/mcp/

Contradiction Tool
curl -N \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": "4",
    "method": "tools/call",
    "params": {
      "name": "detect_contradictions_tool",
      "arguments": {
        "text": "He said he was at home at 8 PM, but was also at the bank at 8 PM."
      }
    }
  }' \
  https://huggingface.co/spaces/MCP-1st-Birthday/sherlock-mcp/gradio_api/mcp/

How to Use Sherlock in the UI
1. Paste or upload text

Emails, reports, statements, stories, chat logs—anything text-based.

2. Select an Investigation Mode

Standard Sherlock

Forensics Mode

Scam Email Mode

Legal / Compliance Mode

3. Submit

Sherlock generates:

Clues

Risk analysis

Timeline

LLM narrative

Optional contradiction/phishing tools

Downloadable report


This project uses:

- OpenAI API
- Track 1 MCP Server (Consumer)
  - Consumer MCP Servers - Tag: "building-mcp-track-consumer"
  - Creative MCP Servers - Tags: "building-mcp-track-creative"
- Track 2 Agentic Application
  - Customer Applications - Tag: "mcp-in-action-track-consumer"
  - Creative Applications - Tag: "mcp-in-action-track-creative"

Solo submission

# 🎓 L&D AI Agent Platform — Demo

An AI-powered **Learning & Development** platform built as a multi-agent system using **Groq LLM**, **FastAPI**, and **Streamlit**. Designed for live demos, the platform automates skill gap analysis, personalized learning paths, compliance tracking, knowledge retention, and on-demand microlearning content generation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agents](#agents)
- [Mock LMS API](#mock-lms-api)
- [Data Model](#data-model)
- [UI Views](#ui-views)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [Environment Variables](#environment-variables)

---

## Overview

The platform simulates an enterprise L&D department powered by AI agents. Each agent is a specialist that handles one domain. A central **Orchestrator** classifies natural-language queries and routes them to the correct agent automatically.

**Key capabilities:**

| Capability | Agent |
|---|---|
| Identify skill gaps vs. competency framework | Skill Gap Analyzer |
| Build a sequenced learning plan from LMS courses | Learning Path Generator |
| Answer course-related questions (RAG-style) | Training Coach |
| Model knowledge decay with Ebbinghaus forgetting curve | Knowledge Retention |
| Track mandatory training, auto-enroll, issue certificates | Compliance Manager |
| Stream a microlearning module live (demo effect) | Content Generator |
| Route any natural-language query to the right agent | Orchestrator |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (port 8501)              │
│  ┌──────────┐  ┌──────────────────────────────────────┐ │
│  │ Sidebar  │  │  View Area (8 views)                 │ │
│  │  Nav     │  │  Orchestrator · Gap · Path · Coach   │ │
│  │          │  │  Retention · Compliance · Content    │ │
│  │          │  │  Generator · Live Demo               │ │
│  └──────────┘  └──────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                            │  Python imports (in-process)
        ┌───────────────────▼───────────────────┐
        │           Agent Layer                 │
        │  ┌─────────────┐  ┌────────────────┐  │
        │  │ Orchestrator│  │ Gap Analyzer   │  │
        │  │             │  │ Learning Path  │  │
        │  │ (intent     │  │ Training Coach │  │
        │  │  router)    │  │ Retention      │  │
        │  └──────┬──────┘  │ Compliance     │  │
        │         │         │ Content Gen    │  │
        │         └─────────┴────────────────┘  │
        └────────┬──────────────────┬────────────┘
                 │                  │
     ┌───────────▼───┐    ┌─────────▼──────────┐
     │   Groq API    │    │  Mock LMS API       │
     │ llama-3.3-70b │    │  FastAPI (port 8000)│
     │  (streaming   │    │  courses, enrollments│
     │   + chat)     │    │  compliance, certs  │
     └───────────────┘    └─────────────────────┘
```

All agents call **Groq** (`llama-3.3-70b-versatile`) for LLM tasks and the **Mock LMS** for operational data (enrollments, completions, compliance records, certificates).

---

## Agents

### 1. Orchestrator (`agents/orchestrator.py`)

The central intent router. Accepts free-text user queries with optional conversation history and classifies them into one of **13 intents**, then returns structured JSON for the UI to dispatch.

**Intents:**

| Intent | Triggers |
|---|---|
| `ANALYZE_EMPLOYEE` | Gap analysis for a named individual |
| `ANALYZE_DEPARTMENT` | Gap analysis for a whole department |
| `ANALYZE_ALL` | Company-wide gap overview |
| `LEARNING_PATH` | Build a learning plan for an employee |
| `COACH` | Conceptual question about course content |
| `RETENTION_EMPLOYEE` | Forgetting curve analysis for one person |
| `RETENTION_ALL` | Team/company-wide retention health |
| `COMPLIANCE_EMPLOYEE` | Compliance status for one person |
| `COMPLIANCE_DEPARTMENT` | Compliance status for a department |
| `COMPLIANCE_ALL` | Company-wide compliance audit |
| `COMPLIANCE_ENROLL` | Auto-enroll non-compliant employees |
| `COMPLIANCE_CERTIFICATES` | Issue or list certificates |
| `UNKNOWN` | Unrecognised query |

Returns: `{"intent": "...", "employee_name": "...", "department": "...", "course_topic": "..."}`

Uses `importlib.reload()` on each call (instead of `@st.cache_resource`) to avoid stale module state across Streamlit reruns.

---

### 2. Skill Gap Analyzer (`agents/gap_analyzer_agent.py`)

Compares an employee's current skills against the **competency framework** for their role and seniority level. Computes a gap percentage and generates an LLM narrative summary.

**Key methods:**
- `analyze_employee(employee_id)` → structured gap dict with `missing_skills`, `present_skills`, `gap_percentage`
- `analyze_department(department)` → aggregated gaps across all employees in a team
- `analyze_all()` → company-wide rollup
- `generate_narrative(data)` → Groq-generated readable summary

---

### 3. Learning Path Generator (`agents/learning_path_agent.py`)

Matches an employee's missing skills to courses in the LMS catalog and builds a sequenced, week-by-week learning schedule. Falls back to the local `courses.json` if the LMS is unavailable.

**Key methods:**
- `generate_path(employee_id, gap_data=None)` → schedule list with `start_week`, `end_week`, `title`, `level`, `skills_covered`
- `generate_narrative(path_data)` → Groq-generated plan description

---

### 4. Training Coach (`agents/coach_agent.py`)

A stateful **RAG-style** conversational coach grounded in course markdown documents. Maintains multi-turn conversation history within a session. Only answers questions from the provided course material.

**Key methods:**
- `chat(user_message)` → LLM reply grounded in course docs
- `select_course(stem)` → narrow context to a specific course
- `reset()` → clear conversation history

**Course documents** (`course_docs/`):
- `python_for_data_teams.md`
- `engineering_leadership_bootcamp.md`

---

### 5. Knowledge Retention Agent (`agents/retention_agent.py`)

Models how well employees retain completed training using the **Ebbinghaus forgetting curve**:

$$R = e^{-t/S}$$

Where $R$ is retention (0–1), $t$ is days since course completion, and $S$ is the stability factor (default: 30 days).

**Urgency thresholds:**

| Score | Label |
|---|---|
| ≥ 70 | `good` — retention healthy |
| 40–69 | `review_soon` — schedule a review |
| < 40 | `review_now` — knowledge at risk |

**Key methods:**
- `analyze_employee(employee_id)` → per-course retention scores + LLM reminder
- `analyze_all()` → team-wide retention health overview
- `generate_reminder(data)` → personalised spaced-repetition nudge
- `generate_team_summary(data)` → Groq-generated team report

---

### 6. Compliance Agent (`agents/compliance_agent.py`)

Manages mandatory training compliance workflows end-to-end:

1. **Check** — compare each employee's completions against `compliance_rules.json`
2. **Enroll** — auto-enroll non-compliant employees via the LMS `/enroll` endpoint
3. **Certify** — issue certificates for newly compliant employees
4. **Audit** — generate a company-wide compliance audit report with LLM narrative

**Key methods:**
- `check_employee(employee_id)` → compliance status + missing mandatory courses
- `check_department(department)` → department-level breakdown
- `check_all()` → company-wide audit data with `overall_compliance_rate_pct`
- `enroll_non_compliant(dept=None)` → batch enrollment
- `issue_certificates_for_compliant()` → batch certificate issuance
- `generate_audit_report(data)` → Groq-generated audit narrative
- `generate_employee_notice(status)` → personalised compliance notice

---

### 7. Content Generator Agent (`agents/content_generator_agent.py`)

Generates a complete microlearning module for any skill gap — **streamed token-by-token** via the Groq streaming API for a live, real-time demo effect.

**Module output format:**
- Title + learning objective
- 3 key concepts with explanations
- 2 quiz questions with multiple-choice answers
- Practical exercise

**Key methods:**
- `generate_stream(skill, role, seniority)` → generator yielding string chunks (use with `st.write_stream` or custom streaming loop)
- `generate(skill, role, seniority)` → full string (non-streaming)

---

## Mock LMS API

A **FastAPI** stub (`mock_lms/server.py`) that simulates a real Learning Management System. All data is in-memory (seeded from JSON files) and resets on server restart.

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/courses` | Full course catalog |
| GET | `/courses/{id}` | Course by ID |
| GET | `/courses/by-skill/{skill}` | Courses covering a skill |
| GET | `/employees` | All employees |
| GET | `/employees/{id}` | Employee by ID |
| POST | `/enroll` | Enroll employee in a course |
| GET | `/enrollment/{employee_id}` | Employee's enrollments |
| GET | `/completion/{employee_id}` | All completions for employee |
| GET | `/completion/{employee_id}/{course_id}` | Single completion record |
| GET | `/retention/{employee_id}` | Retention log for employee |
| GET | `/retention` | All retention records |
| GET | `/compliance/rules` | Mandatory course rules |
| GET | `/compliance/status/{employee_id}` | Compliance status for employee |
| GET | `/compliance/status` | Company-wide compliance status |
| GET | `/certificates` | All issued certificates |
| GET | `/certificates/{employee_id}` | Certificates for employee |
| POST | `/certificates/issue` | Issue a certificate |

---

## Data Model

### Employees (`data/employees.json`)
10 employees (E001–E010) across Engineering, Data, and HR departments.

```json
{
  "id": "E001",
  "name": "Alice Chen",
  "role": "Software Engineer",
  "seniority": "Mid",
  "department": "Engineering",
  "current_skills": ["Python", "JavaScript", "Git", "REST APIs", "SQL"],
  "years_experience": 3
}
```

### Competency Framework (`data/competency_framework.json`)
Required skills per role × seniority level.

**Roles:** Software Engineer, Data Analyst, Team Lead, HR Manager  
**Seniority levels:** Junior, Mid, Senior

### Courses (`data/courses.json`)
12 courses (C001–C012) with skills covered, duration in weeks, and difficulty level.

### Compliance Rules (`data/compliance_rules.json`)
Mandatory courses per role with completion deadline windows.

### Certificates (`data/certificates.json`)
Seed data for 4 pre-issued certificates (CERT-001 to CERT-004).

---

## UI Views

The Streamlit app (`ui/app.py`) provides 8 views selectable from the sidebar:

| View | Description |
|---|---|
| 🏠 **Orchestrator** | Conversational chat — type any HR/L&D question and the system routes it automatically |
| 📊 **Skill Gap Analyzer** | Individual or department gap analysis with visual summaries |
| 🗺️ **Learning Path Generator** | Generate and display a week-by-week learning schedule |
| 💬 **Training Coach** | Multi-turn conversational coach grounded in course documents |
| 🧠 **Knowledge Retention** | Ebbinghaus curve analysis, retention scores, spaced-repetition reminders |
| ✅ **Compliance Manager** | 4-tab view: individual status, department, company audit, certificates |
| ⚡ **Content Generator** | Pick a skill gap → stream a full microlearning module live |
| 🎯 **Live Demo** | Single-screen 360° employee view: heatmap + learning plan + content gen + coach chat |

### Live Demo view layout
```
┌─────────────────────────────────────────────────┐
│            👤 Employee Selector                 │
├─────────────────────┬───────────────────────────┤
│  📊 Skill Heatmap   │   🗺️ Learning Plan Card   │
│  ✅ green badges    │   timeline with levels     │
│  ❌ red badges      │                            │
├─────────────────────┼───────────────────────────┤
│  ⚡ Generate Module │   💬 Coach Chat (embedded) │
│  (streaming)        │   per-employee session     │
└─────────────────────┴───────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Agent framework | LangGraph (StateGraph-based tool orchestration) |
| UI | Streamlit |
| API server | FastAPI + Uvicorn |
| HTTP client | `requests` |
| Environment | `python-dotenv` |
| Python | 3.14 |

---

## Project Structure

```
ld_demo/
├── .env                          # GROQ_API_KEY
├── .env.example
├── start_lms.ps1                 # PowerShell script to start Mock LMS
├── start_ui.ps1                  # PowerShell script to start Streamlit UI
│
├── agents/
│   ├── orchestrator.py           # Intent classifier + router
│   ├── gap_analyzer_agent.py     # Skill gap analysis
│   ├── learning_path_agent.py    # Course sequencing
│   ├── coach_agent.py            # RAG conversational coach
│   ├── retention_agent.py        # Ebbinghaus retention tracker
│   ├── compliance_agent.py       # Mandatory training compliance
│   └── content_generator_agent.py # Streaming microlearning generator
│
├── mock_lms/
│   └── server.py                 # FastAPI LMS stub (port 8000)
│
├── data/
│   ├── employees.json            # 10 employee profiles
│   ├── competency_framework.json # Required skills per role × seniority
│   ├── courses.json              # 12 LMS courses
│   ├── compliance_rules.json     # Mandatory training rules
│   └── certificates.json        # Seed certificates
│
├── course_docs/
│   ├── python_for_data_teams.md          # RAG source for Coach
│   └── engineering_leadership_bootcamp.md # RAG source for Coach
│
└── ui/
    └── app.py                    # Streamlit app (8 views)
```

---

## Setup & Running

### 1. Install dependencies

```powershell
cd ld_demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn streamlit groq python-dotenv requests
pip install langgraph
```

### 2. Configure API key

```powershell
copy .env.example .env
# Edit .env and set your Groq API key:
# GROQ_API_KEY=gsk_...
```

### 3. Start the Mock LMS

```powershell
# From the ld_demo directory
& ".venv\Scripts\python.exe" -m uvicorn mock_lms.server:app --host 127.0.0.1 --port 8000 --app-dir "c:\path\to\ld_demo"
```

Or use the helper script:

```powershell
.\start_lms.ps1
```

### 4. Start the Streamlit UI

```powershell
& ".venv\Scripts\python.exe" -m streamlit run ui\app.py
```

Or use the helper script:

```powershell
.\start_ui.ps1
```

The app will open at **http://localhost:8501**.

> **Note:** The Mock LMS must be running before opening any view that calls it (Learning Path, Retention, Compliance). The Orchestrator, Gap Analyzer, Coach, and Content Generator work without it.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key from [console.groq.com](https://console.groq.com) |

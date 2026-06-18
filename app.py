"""
L&D Multi-Agent Demo — Streamlit UI
Run with: streamlit run ui/app.py
"""
import sys
import os
import json
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="L&D AI Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_gap_agent():
    from agents.gap_analyzer_agent import SkillGapAnalyzerAgent
    return SkillGapAnalyzerAgent()

@st.cache_resource
def get_path_agent():
    from agents.learning_path_agent import LearningPathGeneratorAgent
    return LearningPathGeneratorAgent()

@st.cache_resource
def get_coach_agent():
    from agents.coach_agent import TrainingCoachAgent
    return TrainingCoachAgent()

@st.cache_resource
def get_retention_agent():
    from agents.retention_agent import KnowledgeRetentionAgent
    return KnowledgeRetentionAgent()

@st.cache_resource
def get_compliance_agent():
    from agents.compliance_agent import ComplianceAgent
    return ComplianceAgent()

@st.cache_resource
def get_content_agent():
    from agents.content_generator_agent import ContentGeneratorAgent
    return ContentGeneratorAgent()

@st.cache_resource
def get_agentic_orchestrator():
    from agents.orchestrator import AgenticOrchestrator
    return AgenticOrchestrator()

def load_employees():
    data_path = Path(__file__).parent.parent / "data" / "employees.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)

def gap_color(pct: int) -> str:
    if pct >= 60:
        return "🔴"
    if pct >= 30:
        return "🟡"
    return "🟢"

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 L&D AI Platform")
    st.caption("Powered by Groq · Llama 3.3 70B")
    st.divider()
    view = st.radio(
        "Navigate",
        [
            "🏠 Orchestrator",
            "📊 Skill Gap Analyzer",
            "🗺️ Learning Path Generator",
            "💬 Training Coach",
            "🧠 Knowledge Retention",
            "✅ Compliance Manager",
            "⚡ Content Generator",
            "🎯 Live Demo",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Mock data · 10 employees · 12 courses")

# ── Orchestrator View ──────────────────────────────────────────────────────
if view == "🏠 Orchestrator":
    st.title("🏠 L&D Orchestrator")
    st.caption(
        "Agentic AI — the LLM autonomously selects, sequences, and chains specialist tools. "
        "Conversation history persists across sessions via SQLite."
    )

    from agents.memory import init_db, new_session, list_sessions, load_history as load_mem_history
    init_db()

    # ── Session management ─────────────────────────────────────────────────
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = new_session()
        st.session_state["loaded_session_id"] = None

    col_sess, col_new = st.columns([4, 1])
    sessions = list_sessions()
    session_map = {
        f"{s['label']}  ({s['last_active'][:10]})": s["session_id"]
        for s in sessions
    }

    with col_sess:
        if session_map:
            options = ["▶ Keep current session"] + list(session_map.keys())
            picked_label = st.selectbox(
                "Load a past session",
                options,
                key="session_picker",
                label_visibility="collapsed",
            )
            if picked_label != "▶ Keep current session":
                picked_id = session_map[picked_label]
                if picked_id != st.session_state["current_session_id"]:
                    st.session_state["current_session_id"] = picked_id
                    st.session_state["loaded_session_id"] = None
                    st.rerun()

    with col_new:
        if st.button("＋ New", use_container_width=True, key="new_session_btn"):
            st.session_state["current_session_id"] = new_session()
            st.session_state["loaded_session_id"] = None
            st.rerun()

    session_id = st.session_state["current_session_id"]
    st.caption(f"Session `{session_id}`")

    # ── Load conversation from SQLite when session changes ─────────────────
    if st.session_state.get("loaded_session_id") != session_id:
        history_rows = load_mem_history(session_id, limit=30)
        st.session_state["orch_messages"] = [
            {"role": h["role"], "content": h["content"], "tools_used": h.get("tools_used")}
            for h in history_rows
        ]
        st.session_state["loaded_session_id"] = session_id

    orch = get_agentic_orchestrator()

    # ── Suggested prompts (only shown on empty sessions) ───────────────────
    if not st.session_state["orch_messages"]:
        st.markdown("**Try asking:**")
        examples = [
            "Analyze gaps for Bob Martinez then build him a learning plan",
            "What are the top skill gaps across the Engineering department?",
            "Is the HR department compliant? If not, enroll them.",
            "Show me company-wide retention health",
            "What is spaced repetition and why does it improve training ROI?",
        ]
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["orch_pending"] = ex
                st.rerun()

    # ── Render persisted conversation ──────────────────────────────────────
    for msg in st.session_state["orch_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                st.caption("🔧 " + " → ".join(msg["tools_used"]))

    # ── Clear button ───────────────────────────────────────────────────────
    if st.session_state["orch_messages"]:
        if st.button("🗑️ Clear conversation", key="clear_orch"):
            st.session_state["current_session_id"] = new_session()
            st.session_state["loaded_session_id"] = None
            st.rerun()

    # ── Handle suggested-prompt clicks ─────────────────────────────────────
    pending = st.session_state.pop("orch_pending", None)

    # ── Chat input ─────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about gaps, learning paths, compliance, or course topics…") or pending

    if user_input:
        st.session_state["orch_messages"].append(
            {"role": "user", "content": user_input, "tools_used": None}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            status = st.empty()
            answer_placeholder = st.empty()
            tool_trace: list[dict] = []
            final_answer = ""
            tools_used: list[str] = []

            for event in orch.run(user_input, session_id):
                if event["type"] == "tool_call":
                    label = event["tool"].replace("_", " ").title()
                    args_str = ", ".join(f"{k}={v}" for k, v in event["args"].items())
                    status.info(f"🔧 Calling **{label}**({args_str})…")

                elif event["type"] == "tool_result":
                    tool_trace.append(event)

                elif event["type"] == "answer_delta":
                    final_answer += event.get("delta", "")
                    answer_placeholder.markdown(final_answer)

                elif event["type"] == "answer":
                    status.empty()
                    if not final_answer:
                        final_answer = event["content"]
                    tools_used = event.get("tools_used", [])
                    answer_placeholder.markdown(final_answer)

            if tools_used:
                st.caption("🔧 " + " → ".join(tools_used))

            if tool_trace:
                with st.expander(
                    f"🔍 Agent reasoning trace ({len(tool_trace)} tool call(s))",
                    expanded=False,
                ):
                    for t in tool_trace:
                        st.markdown(f"**`{t['tool']}`**")
                        try:
                            st.json(json.loads(t["result"]), expanded=False)
                        except Exception:
                            st.code(t["result"][:500])

        st.session_state["orch_messages"].append(
            {"role": "assistant", "content": final_answer, "tools_used": tools_used}
        )


# ── Skill Gap Analyzer View ────────────────────────────────────────────────
elif view == "📊 Skill Gap Analyzer":
    st.title("📊 Skill Gap Analyzer")

    employees = load_employees()
    departments = sorted(set(e["department"] for e in employees))

    tab1, tab2, tab3 = st.tabs(["Individual", "Department", "Company-wide"])

    with tab1:
        emp_options = {f"{e['name']} ({e['role']}, {e['seniority']})": e["id"] for e in employees}
        selected = st.selectbox("Select Employee", list(emp_options.keys()))
        if st.button("Analyze", key="gap_emp"):
            emp_id = emp_options[selected]
            gap_agent = get_gap_agent()
            with st.spinner("Analyzing…"):
                gap = gap_agent.analyze_employee(emp_id)
                narrative = gap_agent.generate_narrative(gap)

            col1, col2, col3 = st.columns(3)
            col1.metric("Gap", f"{gap['gap_percentage']}% {gap_color(gap['gap_percentage'])}")
            col2.metric("Missing", gap["skills_missing_count"])
            col3.metric("Required", gap["skills_total_required"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### ✅ Present Skills")
                for s in gap["present_skills"]:
                    st.write(f"- {s}")
            with col_b:
                st.markdown("#### ❌ Missing Skills")
                for s in gap["missing_skills"]:
                    st.write(f"- {s}")

            st.markdown("#### AI Narrative")
            st.markdown(narrative)

    with tab2:
        dept = st.selectbox("Select Department", departments)
        if st.button("Analyze Department", key="gap_dept"):
            gap_agent = get_gap_agent()
            with st.spinner(f"Analyzing {dept}…"):
                data = gap_agent.analyze_department(dept)
                narrative = gap_agent.generate_narrative(data)

            st.metric("Average Gap", f"{data['average_gap_percentage']}%")
            st.markdown("#### Top Missing Skills Across Team")
            for skill, count in data["top_missing_skills"]:
                pct = round(count / data["employee_count"] * 100)
                st.progress(pct / 100, text=f"{skill}  —  {count}/{data['employee_count']} employees")

            st.markdown("#### Individual Gaps")
            for g in sorted(data["individual_gaps"], key=lambda x: -x["gap_percentage"]):
                st.write(
                    f"{gap_color(g['gap_percentage'])} **{g['name']}** ({g['seniority']}) — "
                    f"{g['gap_percentage']}% gap · missing: {', '.join(g['missing_skills']) or 'none'}"
                )

            st.markdown("#### AI Analysis")
            st.markdown(narrative)

    with tab3:
        if st.button("Run Company-wide Analysis"):
            gap_agent = get_gap_agent()
            with st.spinner("Analyzing all employees…"):
                data = gap_agent.analyze_all()
                narrative = gap_agent.generate_narrative(data)

            st.metric("Total Employees Analyzed", data["total_employees"])
            for dept_name, summary in data["department_summaries"].items():
                with st.expander(f"**{dept_name}**  —  avg gap {summary['avg_gap_pct']}%"):
                    for skill, count in summary["top_missing_skills"]:
                        st.write(f"- {skill} ({count} employees)")

            st.markdown("#### AI Analysis")
            st.markdown(narrative)


# ── Learning Path Generator View ───────────────────────────────────────────
elif view == "🗺️ Learning Path Generator":
    st.title("🗺️ Learning Path Generator")

    employees = load_employees()
    emp_options = {f"{e['name']} ({e['role']}, {e['seniority']})": e["id"] for e in employees}
    selected = st.selectbox("Select Employee", list(emp_options.keys()))

    if st.button("Generate Learning Path", type="primary"):
        emp_id = emp_options[selected]
        path_agent = get_path_agent()

        with st.spinner("Building personalised learning path…"):
            path = path_agent.generate_path(emp_id)
            narrative = path_agent.generate_narrative(path)

        if "error" in path:
            st.error(path["error"])
        elif not path["schedule"]:
            st.success(path["message"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Courses", path["total_courses"])
            col2.metric("Total Duration", f"{path['total_weeks']} weeks")
            col3.metric("Skills to Gain", len(path["covered_by_courses"]))

            st.markdown("#### 📅 Course Schedule")
            for item in path["schedule"]:
                with st.expander(
                    f"Weeks {item['start_week']}–{item['end_week']}  ·  {item['title']}  [{item['level']}]"
                ):
                    st.write(f"**Duration:** {item['duration_weeks']} weeks")
                    st.write(f"**Skills covered:** {', '.join(item['skills_covered'])}")

            if path["not_covered_by_courses"]:
                st.warning(
                    f"**Skills not yet covered by available courses:** "
                    f"{', '.join(path['not_covered_by_courses'])}"
                )

            st.markdown("#### 📝 Personalised Development Plan")
            st.markdown(narrative)


# ── Training Coach View ────────────────────────────────────────────────────
elif view == "💬 Training Coach":
    st.title("💬 Training Coach")
    st.markdown("Ask questions about your course material. The coach uses the actual course documents to answer.")

    coach = get_coach_agent()
    available = coach.available_courses()

    course_labels = {c.replace("_", " ").title(): c for c in available}
    course_labels["🌐 All courses"] = None

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_label = st.selectbox("Focus on a course", list(course_labels.keys()))
    with col2:
        if st.button("Clear chat"):
            coach.reset()
            st.session_state["coach_history"] = []
            st.rerun()

    selected_course = course_labels[selected_label]
    result = coach.select_course(selected_course) if selected_course else "All course documents loaded."
    st.caption(result)

    # Session state for chat history display
    if "coach_history" not in st.session_state:
        st.session_state["coach_history"] = []

    # Display history
    for msg in st.session_state["coach_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_msg := st.chat_input("Ask your coach a question…"):
        st.session_state["coach_history"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = coach.chat(user_msg)
            st.markdown(reply)

        st.session_state["coach_history"].append({"role": "assistant", "content": reply})


# ── Knowledge Retention View ───────────────────────────────────────────────
elif view == "🧠 Knowledge Retention":
    st.title("🧠 Knowledge Retention Tracker")
    st.markdown(
        "Tracks how well employees retain completed course material using the "
        "**Ebbinghaus forgetting curve**. Identifies at-risk learners and generates review reminders."
    )

    employees = load_employees()
    retention_agent = get_retention_agent()

    tab1, tab2 = st.tabs(["Individual", "Team Overview"])

    with tab1:
        emp_options = {f"{e['name']} ({e['role']})": e["id"] for e in employees}
        selected = st.selectbox("Select Employee", list(emp_options.keys()), key="ret_emp")
        if st.button("Analyse Retention", type="primary", key="ret_btn"):
            emp_id = emp_options[selected]
            with st.spinner("Calculating retention scores…"):
                data = retention_agent.analyze_employee(emp_id)

            if "message" in data:
                st.info(data["message"])
            elif "error" in data:
                st.error(data["error"])
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Retention", f"{data['average_retention_score']}%")
                col2.metric("Courses Tracked", data["courses_tracked"])
                col3.metric("Courses At Risk", data["courses_at_risk"])

                st.markdown("#### Course Retention Scores")
                for c in data["course_scores"]:
                    urgency_icon = {"good": "🟢", "review_soon": "🟡", "review_now": "🔴"}.get(c["urgency"], "⚪")
                    st.progress(
                        c["retention_score"] / 100,
                        text=f"{urgency_icon} **{c['course_title']}** — {c['retention_score']}% retained · completed {c['days_elapsed']} days ago",
                    )

                with st.spinner("Generating personalised reminder…"):
                    reminder = retention_agent.generate_reminder(data)
                st.markdown("#### 📬 Personalised Review Reminder")
                st.markdown(reminder)

    with tab2:
        if st.button("Analyse Team Retention", key="ret_team_btn"):
            with st.spinner("Analysing team retention…"):
                data = retention_agent.analyze_all()
                summary = retention_agent.generate_team_summary(data)

            col1, col2 = st.columns(2)
            col1.metric("Employees Tracked", data["total_tracked"])
            col2.metric("Employees At Risk", data["employees_at_risk"])

            st.markdown("#### Individual Retention Scores")
            for r in data["results"]:
                health = "🔴" if r["courses_at_risk"] > 0 else "🟢"
                with st.expander(f"{health} **{r['name']}** ({r['role']}) — avg {r['average_retention_score']}% · {r['courses_at_risk']} at risk"):
                    for c in r["course_scores"]:
                        urgency_icon = {"good": "🟢", "review_soon": "🟡", "review_now": "🔴"}.get(c["urgency"], "⚪")
                        st.write(f"{urgency_icon} {c['course_title']} — {c['retention_score']}% ({c['days_elapsed']}d ago)")

            st.markdown("#### AI Team Summary")
            st.markdown(summary)


# ── Compliance Manager View ────────────────────────────────────────────────
elif view == "✅ Compliance Manager":
    st.title("✅ Compliance Manager")
    st.markdown("Track mandatory training compliance, auto-enroll non-compliant employees, and issue certificates.")

    employees = load_employees()
    departments = sorted(set(e["department"] for e in employees))
    compliance_agent = get_compliance_agent()

    tab1, tab2, tab3, tab4 = st.tabs(["Individual", "Department", "Company Audit", "Certificates"])

    with tab1:
        emp_options = {f"{e['name']} ({e['role']})": e["id"] for e in employees}
        selected = st.selectbox("Select Employee", list(emp_options.keys()), key="comp_emp")
        if st.button("Check Compliance", type="primary", key="comp_emp_btn"):
            emp_id = emp_options[selected]
            with st.spinner("Checking compliance…"):
                data = compliance_agent.check_employee(emp_id)

            status_icon = "✅ Fully Compliant" if data["fully_compliant"] else "❌ Non-Compliant"
            st.subheader(f"{data['name']} — {status_icon}")
            col1, col2 = st.columns(2)
            col1.metric("Required Courses", data["required_courses"])
            col2.metric("Completed", data["compliant_courses"])

            for cs in data["course_statuses"]:
                icon = "✅" if cs["compliant"] else "❌"
                st.write(f"{icon} **{cs['course_name']}** — {cs['completion_pct']}% complete")

            if not data["fully_compliant"]:
                with st.spinner("Generating compliance notice…"):
                    notice = compliance_agent.generate_employee_notice(data)
                st.markdown("#### 📋 Compliance Notice")
                st.markdown(notice)

    with tab2:
        dept = st.selectbox("Select Department", departments, key="comp_dept")
        col_a, col_b = st.columns(2)
        run_check = col_a.button("Check Compliance", key="comp_dept_btn")
        run_enroll = col_b.button("Auto-Enroll Non-Compliant", key="comp_enroll_btn")

        if run_check:
            with st.spinner(f"Checking {dept} compliance…"):
                data = compliance_agent.check_department(dept)
                if "error" in data:
                    st.error(data["error"])
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Compliance Rate", f"{data['compliance_rate_pct']}%")
                    col2.metric("Fully Compliant", data["fully_compliant"])
                    col3.metric("Non-Compliant", data["non_compliant_count"])

                    for s in sorted(data["all_statuses"], key=lambda x: x["fully_compliant"]):
                        icon = "✅" if s["fully_compliant"] else "❌"
                        missing = [cs["course_name"] for cs in s["course_statuses"] if not cs["compliant"]]
                        st.write(f"{icon} **{s['name']}**" + (f" — missing: {', '.join(missing)}" if missing else ""))

                    with st.spinner("Generating audit report…"):
                        report = compliance_agent.generate_audit_report(data)
                    st.markdown("#### AI Audit Report")
                    st.markdown(report)

        if run_enroll:
            with st.spinner(f"Enrolling non-compliant employees in {dept}…"):
                result = compliance_agent.enroll_non_compliant(dept)
            st.success(f"**{result['enrollments_made']} enrollment(s) made**")
            for a in result["actions"]:
                st.write(f"- **{a['employee']}** → {a['course']} ({a['enrollment_result']})")
            if not result["actions"]:
                st.info("All employees in this department are already enrolled.")

    with tab3:
        if st.button("Run Company-wide Audit", type="primary", key="comp_all_btn"):
            with st.spinner("Running compliance audit…"):
                data = compliance_agent.check_all()
                report = compliance_agent.generate_audit_report(data)

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Rate", f"{data['overall_compliance_rate_pct']}%")
            col2.metric("Fully Compliant", data["fully_compliant"])
            col3.metric("Non-Compliant", data["non_compliant"])

            st.markdown("#### Department Breakdown")
            for dept_name, s in data["by_department"].items():
                color = "🟢" if s["rate_pct"] >= 80 else ("🟡" if s["rate_pct"] >= 50 else "🔴")
                st.progress(s["rate_pct"] / 100, text=f"{color} **{dept_name}** — {s['rate_pct']}% ({s['compliant']}/{s['total']})")

            st.markdown("#### AI Audit Report")
            st.markdown(report)

            col_enroll, col_cert = st.columns(2)
            if col_enroll.button("⚡ Auto-Enroll All Non-Compliant", key="enroll_all"):
                with st.spinner("Enrolling…"):
                    enroll_result = compliance_agent.enroll_non_compliant()
                st.success(f"{enroll_result['enrollments_made']} enrollment(s) made")

            if col_cert.button("🎓 Issue All Pending Certificates", key="cert_all"):
                with st.spinner("Issuing certificates…"):
                    cert_result = compliance_agent.issue_certificates_for_compliant()
                st.success(f"{cert_result['certificates_issued']} certificate(s) issued")

    with tab4:
        if st.button("Refresh Certificates", key="cert_list_btn"):
            certs = compliance_agent.issue_certificates_for_compliant()
            if certs["certificates_issued"] > 0:
                st.success(f"Issued {certs['certificates_issued']} new certificate(s)")

        import requests as _req
        all_certs = _req.get("http://localhost:8000/certificates", timeout=5).json()
        if all_certs:
            st.markdown(f"**{len(all_certs)} certificate(s) on record**")
            for c in sorted(all_certs, key=lambda x: x["issued_date"], reverse=True):
                status_icon = {"valid": "🟢", "expiring_soon": "🟡", "expired": "🔴"}.get(c["status"], "⚪")
                st.write(f"{status_icon} **{c['employee_name']}** — {c['course_name']} · issued {c['issued_date']} · expires {c['expires_date']}")
        else:
            st.info("No certificates on record yet. Run an audit to issue certificates.")


# ── Content Generator View ─────────────────────────────────────────────────
elif view == "⚡ Content Generator":
    st.title("⚡ Microlearning Content Generator")
    st.markdown(
        "Select an employee's skill gap and watch a full microlearning module "
        "**generate in real-time** — title, key concepts, quiz questions, and a practical exercise."
    )

    employees = load_employees()
    gap_agent = get_gap_agent()
    content_agent = get_content_agent()

    emp_options = {f"{e['name']} ({e['role']}, {e['seniority']})": e for e in employees}
    selected_label = st.selectbox("Select Employee", list(emp_options.keys()), key="cg_emp")
    selected_emp = emp_options[selected_label]

    # Compute gaps for this employee to populate skill picker
    gap_data = gap_agent.analyze_employee(selected_emp["id"])
    missing = gap_data.get("missing_skills", [])

    if not missing:
        st.success(f"{selected_emp['name']} has no skill gaps — fully proficient for their role!")
    else:
        skill = st.selectbox(
            "Choose a skill gap to generate a module for",
            missing,
            key="cg_skill",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption(f"Role: **{selected_emp['role']}** · Seniority: **{selected_emp['seniority']}**")
        with col2:
            generate = st.button("⚡ Generate Module", type="primary", key="cg_btn", use_container_width=True)

        if generate:
            st.divider()
            st.markdown(f"*Generating microlearning module for **{skill}**…*")
            output_area = st.empty()
            full_text = ""
            with st.spinner(""):
                for chunk in content_agent.generate_stream(skill, selected_emp["role"], selected_emp["seniority"]):
                    full_text += chunk
                    output_area.markdown(full_text + "▌")  # blinking cursor effect
            output_area.markdown(full_text)  # final render without cursor

            st.divider()
            col_a, col_b = st.columns(2)
            col_a.success("✅ Module generated successfully")
            if col_b.button("🔁 Generate Another Skill", key="cg_another"):
                st.rerun()


# ── Live Demo View ─────────────────────────────────────────────────────────
elif view == "🎯 Live Demo":
    st.title("🎯 Live Demo — Employee 360° View")
    st.caption("Single-screen snapshot: select an employee to see their full learning profile instantly.")

    employees = load_employees()
    gap_agent = get_gap_agent()
    path_agent = get_path_agent()
    coach = get_coach_agent()

    # ── Employee selector ──────────────────────────────────────────────────
    emp_options = {f"{e['name']}  ·  {e['role']} ({e['seniority']})": e for e in employees}
    selected_label = st.selectbox(
        "👤 Select Employee",
        list(emp_options.keys()),
        key="demo_emp",
        label_visibility="visible",
    )
    emp = emp_options[selected_label]
    emp_id = emp["id"]

    st.divider()

    # ── Row 1: gap heatmap + learning path card ────────────────────────────
    col_gap, col_path = st.columns(2, gap="large")

    with col_gap:
        st.markdown("### 📊 Skill Heatmap")
        with st.spinner("Loading gaps…"):
            gap_data = gap_agent.analyze_employee(emp_id)

        # Render skills as coloured badges using HTML
        badge_html = "<div style='line-height:2.2;'>"
        for skill in gap_data["required_skills"]:
            if skill in gap_data["missing_skills"]:
                badge_html += (
                    f"<span style='background:#ff4b4b;color:white;padding:3px 10px;"
                    f"border-radius:12px;margin:3px;display:inline-block;font-size:13px;'>"
                    f"❌ {skill}</span>"
                )
            else:
                badge_html += (
                    f"<span style='background:#21c354;color:white;padding:3px 10px;"
                    f"border-radius:12px;margin:3px;display:inline-block;font-size:13px;'>"
                    f"✅ {skill}</span>"
                )
        badge_html += "</div>"
        st.html(badge_html)

        gap_pct = gap_data["gap_percentage"]
        color = "#ff4b4b" if gap_pct >= 60 else ("#ffa421" if gap_pct >= 30 else "#21c354")
        st.markdown(
            f"<p style='font-size:15px;margin-top:8px;'>"
            f"Gap: <strong style='color:{color}'>{gap_pct}%</strong> · "
            f"Missing <strong>{gap_data['skills_missing_count']}</strong> of "
            f"<strong>{gap_data['skills_total_required']}</strong> required skills</p>",
            unsafe_allow_html=True,
        )

    with col_path:
        st.markdown("### 🗺️ Learning Plan")
        with st.spinner("Building path…"):
            path_data = path_agent.generate_path(emp_id, gap_data)

        if not path_data.get("schedule"):
            st.success("No gaps — this employee is fully up to speed! 🎉")
        else:
            # Compact timeline card
            for item in path_data["schedule"]:
                level_color = {"Beginner": "#1c83e1", "Intermediate": "#ffa421", "Advanced": "#ff4b4b"}.get(
                    item["level"], "#888"
                )
                st.markdown(
                    f"<div style='border-left:4px solid {level_color};padding:6px 12px;margin-bottom:6px;"
                    f"background:#f8f9fa;border-radius:0 8px 8px 0;'>"
                    f"<strong>Wk {item['start_week']}–{item['end_week']}</strong> &nbsp;"
                    f"<span style='color:{level_color};font-size:12px;'>[{item['level']}]</span><br/>"
                    f"{item['title']}<br/>"
                    f"<span style='font-size:12px;color:#666;'>{', '.join(item['skills_covered'])}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.caption(f"Total: **{path_data['total_courses']} courses** · **{path_data['total_weeks']} weeks**")

    st.divider()

    # ── Row 2: Content generator + Coach chat ────────────────────────────
    col_content, col_coach = st.columns(2, gap="large")

    with col_content:
        st.markdown("### ⚡ Generate Microlearning")
        missing_skills = gap_data.get("missing_skills", [])
        if not missing_skills:
            st.info("No skill gaps to generate content for.")
        else:
            skill_pick = st.selectbox("Pick a skill gap", missing_skills, key="demo_skill")
            if st.button("⚡ Generate", type="primary", key="demo_gen", use_container_width=True):
                content_agent = get_content_agent()
                output_area = st.empty()
                full_text = ""
                for chunk in content_agent.generate_stream(skill_pick, emp["role"], emp["seniority"]):
                    full_text += chunk
                    output_area.markdown(full_text + "▌")
                output_area.markdown(full_text)

    with col_coach:
        st.markdown("### 💬 Training Coach")
        st.caption("Ask anything about the employee's assigned courses.")

        # Per-employee coach session in demo view
        session_key = f"demo_coach_{emp_id}"
        if session_key not in st.session_state:
            st.session_state[session_key] = []

        # Display chat history
        chat_container = st.container(height=320)
        with chat_container:
            for msg in st.session_state[session_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if demo_msg := st.chat_input("Ask the coach…", key=f"demo_chat_{emp_id}"):
            st.session_state[session_key].append({"role": "user", "content": demo_msg})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(demo_msg)
                with st.chat_message("assistant"):
                    with st.spinner(""):
                        reply = coach.chat(demo_msg)
                    st.markdown(reply)
            st.session_state[session_key].append({"role": "assistant", "content": reply})
            st.rerun()

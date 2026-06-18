"""
Knowledge Retention Agent
Uses the Ebbinghaus forgetting curve to model how well employees retain
completed course material over time, and generates personalised review reminders.

Forgetting curve: R = e^(-t/S)
  R = retention (0–1), t = days since completion, S = stability factor (default 30)
"""
import json
import math
import os
from datetime import date, datetime
from pathlib import Path

import requests
from groq import Groq

DATA_DIR = Path(__file__).parent.parent / "data"
LMS_BASE = "http://localhost:8000"


def _load(filename: str):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def _get(path: str):
    try:
        r = requests.get(f"{LMS_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _retention_score(completed_date_str: str, stability: int = 30) -> dict:
    """
    Compute Ebbinghaus retention score for a course completion.
    Returns score (0–100), days elapsed, and urgency label.
    """
    completed = datetime.strptime(completed_date_str, "%Y-%m-%d").date()
    days_elapsed = (date.today() - completed).days
    score = round(math.exp(-days_elapsed / stability) * 100)

    if score >= 70:
        urgency = "good"
    elif score >= 40:
        urgency = "review_soon"
    else:
        urgency = "review_now"

    # Spaced repetition: next review interval doubles each time (1→2→4→8 weeks)
    review_interval = max(7, stability - days_elapsed % stability)
    next_review = date.today().isoformat()  # simplification for demo

    return {
        "completed_date": completed_date_str,
        "days_elapsed": days_elapsed,
        "retention_score": score,
        "urgency": urgency,
        "next_review_in_days": review_interval,
    }


class KnowledgeRetentionAgent:
    """
    Analyses knowledge retention for employees using the Ebbinghaus forgetting curve.
    Identifies at-risk learners and generates personalised review reminders.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.employees = _load("employees.json")
        self.courses = {c["id"]: c for c in _load("courses.json")}

    def _get_retention_log(self, employee_id: str) -> list[dict]:
        data = _get(f"/retention/{employee_id}")
        return data["history"] if data else []

    def analyze_employee(self, employee_id: str) -> dict:
        """Return retention scores for all completed courses for one employee."""
        emp = next((e for e in self.employees if e["id"] == employee_id), None)
        if not emp:
            print(f"[Retention] ✗ Employee '{employee_id}' not found")
            return {"error": f"Employee {employee_id} not found"}

        print(f"[Retention] ▶ Analyzing knowledge retention for {emp['name']} ({emp['role']})...")        
        history = self._get_retention_log(employee_id)
        if not history:
            print(f"[Retention]   No completed courses found for {emp['name']}")
            return {
                "employee_id": employee_id,
                "name": emp["name"],
                "message": "No completed courses found — nothing to track yet.",
                "courses": [],
            }

        course_scores = []
        for entry in history:
            cid = entry["course_id"]
            course = self.courses.get(cid, {})
            score_data = _retention_score(entry["completed_date"])
            course_scores.append({
                "course_id": cid,
                "course_title": course.get("title", cid),
                **score_data,
            })

        # Sort by retention score ascending (worst first)
        course_scores.sort(key=lambda x: x["retention_score"])
        avg_retention = round(sum(c["retention_score"] for c in course_scores) / len(course_scores))
        at_risk = [c for c in course_scores if c["urgency"] != "good"]

        print(f"[Retention] ✓ {emp['name']}: avg retention {avg_retention}%, {len(at_risk)}/{len(course_scores)} courses need review")
        for c in at_risk:
            print(f"[Retention]   ⚠ '{c['course_title']}': {c['retention_score']}% retained ({c['days_elapsed']}d ago) — {c['urgency']}")

        return {
            "employee_id": employee_id,
            "name": emp["name"],
            "role": emp["role"],
            "average_retention_score": avg_retention,
            "courses_tracked": len(course_scores),
            "courses_at_risk": len(at_risk),
            "course_scores": course_scores,
        }

    def analyze_all(self) -> dict:
        """Return retention analysis for every employee who has completed courses."""
        print("[Retention] ▶ Running company-wide retention analysis...")
        all_logs = _get("/retention") or {}
        results = []

        for emp_id, history in all_logs.items():
            if history:
                result = self.analyze_employee(emp_id)
                if "error" not in result:
                    results.append(result)

        results.sort(key=lambda x: x["average_retention_score"])

        at_risk_employees = [r for r in results if r["courses_at_risk"] > 0]
        print(f"[Retention] ✓ Team retention: {len(results)} employees tracked, {len(at_risk_employees)} at risk")
        return {
            "total_tracked": len(results),
            "employees_at_risk": len(at_risk_employees),
            "results": results,
        }

    def generate_reminder(self, employee_data: dict) -> str:
        """Use Groq LLM to write a personalised review reminder for an employee."""
        at_risk = [c for c in employee_data.get("course_scores", []) if c["urgency"] != "good"]
        if not at_risk:
            print(f"[Retention] ✓ {employee_data['name']} has strong retention — no reminder needed")
            return f"Great news! {employee_data['name']} has strong retention across all completed courses."

        print(f"[Retention] ▶ Generating review reminder for {employee_data['name']} ({len(at_risk)} courses at risk)...")

        prompt = f"""You are an L&D coach writing a personalised knowledge review reminder.

Write a short, encouraging message (2-3 paragraphs) to the employee letting them know:
- Which course(s) they should review based on retention data
- Why reviewing now will help them (link to their role)
- One specific thing they can do today (re-read a section, take a quiz, etc.)

Employee: {employee_data['name']} ({employee_data['role']})
Average retention: {employee_data['average_retention_score']}%
Courses needing review:
{json.dumps(at_risk, indent=2)}

Tone: friendly, motivating, not alarming. Address them by first name."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
        )
        reminder = response.choices[0].message.content or ""
        print(f"[Retention] ✓ Reminder generated for {employee_data['name']} ({len(reminder)} chars)")
        return reminder

    def generate_team_summary(self, data: dict) -> str:
        """Generate an L&D manager summary of team-wide retention health."""
        print("[Retention] ▶ Generating team retention summary via LLM...")
        prompt = f"""You are an L&D analyst writing a retention health summary for an HR manager.

Summarise the team's knowledge retention status in 3-4 paragraphs:
- Overall retention health
- Employees who need immediate review attention
- Recommended actions (team review sessions, nudges, spaced repetition schedule)

Data:
{json.dumps(data, indent=2)}

Be concise and action-oriented."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        summary = response.choices[0].message.content or ""
        print(f"[Retention] ✓ Team summary generated ({len(summary)} chars)")
        return summary

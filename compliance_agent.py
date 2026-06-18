"""
Compliance Training Orchestrator Agent
Manages mandatory training workflows:
  1. Check compliance status per employee / department / company
  2. Auto-enroll non-compliant employees
  3. Issue certificates for newly compliant employees
  4. Generate audit reports with human-in-the-loop approval gate
"""
import json
import os
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


def _post(path: str, payload: dict):
    try:
        r = requests.post(f"{LMS_BASE}{path}", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


class ComplianceAgent:
    """
    Orchestrates mandatory training compliance across the organisation.
    Supports checking status, auto-enrolling, issuing certificates, and generating audit reports.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.employees = _load("employees.json")
        self.rules = _load("compliance_rules.json")

    # ── Status checks ────────────────────────────────────────────────────────

    def check_employee(self, employee_id: str) -> dict:
        """Return compliance status for a single employee."""
        print(f"[Compliance] ▶ Checking compliance status for employee '{employee_id}'...")
        data = _get(f"/compliance/status/{employee_id}")
        if not data:
            print(f"[Compliance] ✗ Could not fetch compliance data for '{employee_id}'")
            return {"error": f"Could not fetch compliance for {employee_id}"}
        status = "✅ Fully compliant" if data.get("fully_compliant") else "⚠ Non-compliant"
        print(f"[Compliance] ✓ {data.get('name', employee_id)}: {status}")
        return data

    def check_department(self, department: str) -> dict:
        """Return compliance status for all employees in a department."""
        print(f"[Compliance] ▶ Checking compliance for department '{department}'...")
        all_statuses = _get("/compliance/status") or []
        dept_statuses = [
            s for s in all_statuses
            if s.get("department", "").lower() == department.lower()
        ]
        if not dept_statuses:
            print(f"[Compliance] ✗ No employees found in '{department}'")
            return {"error": f"No employees found in department '{department}'"}

        total = len(dept_statuses)
        fully_compliant = sum(1 for s in dept_statuses if s["fully_compliant"])
        non_compliant = [s for s in dept_statuses if not s["fully_compliant"]]

        print(f"[Compliance] ✓ {department}: {fully_compliant}/{total} compliant ({round(fully_compliant / total * 100)}%)")
        if non_compliant:
            print(f"[Compliance]   Non-compliant: {', '.join(s['name'] for s in non_compliant)}")

        return {
            "department": department,
            "total_employees": total,
            "fully_compliant": fully_compliant,
            "non_compliant_count": total - fully_compliant,
            "compliance_rate_pct": round(fully_compliant / total * 100),
            "non_compliant_employees": non_compliant,
            "all_statuses": dept_statuses,
        }

    def check_all(self) -> dict:
        """Return company-wide compliance overview."""
        print("[Compliance] ▶ Running company-wide compliance audit...")
        all_statuses = _get("/compliance/status") or []
        total = len(all_statuses)
        if total == 0:
            print("[Compliance] ✗ No compliance data available")
            return {"error": "No compliance data available"}

        fully_compliant = sum(1 for s in all_statuses if s["fully_compliant"])
        print(f"[Compliance] ✓ Company-wide: {fully_compliant}/{total} compliant ({round(fully_compliant / total * 100)}%)")
        by_dept: dict[str, dict] = {}
        for s in all_statuses:
            dept = s.get("department", "Unknown")
            if dept not in by_dept:
                by_dept[dept] = {"total": 0, "compliant": 0}
            by_dept[dept]["total"] += 1
            if s["fully_compliant"]:
                by_dept[dept]["compliant"] += 1

        dept_summary = {
            dept: {
                "total": v["total"],
                "compliant": v["compliant"],
                "rate_pct": round(v["compliant"] / v["total"] * 100),
            }
            for dept, v in by_dept.items()
        }

        return {
            "total_employees": total,
            "fully_compliant": fully_compliant,
            "non_compliant": total - fully_compliant,
            "overall_compliance_rate_pct": round(fully_compliant / total * 100),
            "by_department": dept_summary,
            "all_statuses": all_statuses,
        }

    # ── Actions ──────────────────────────────────────────────────────────────

    def enroll_non_compliant(self, department: str | None = None) -> dict:
        """
        Auto-enroll all non-compliant employees in their missing mandatory courses.
        Returns a list of enrollment actions taken (human-in-the-loop: shown for approval).
        """
        scope = f"'{department}' department" if department else "all departments"
        print(f"[Compliance] ▶ Auto-enrolling non-compliant employees ({scope})...")
        all_statuses = _get("/compliance/status") or []
        if department:
            all_statuses = [
                s for s in all_statuses
                if s.get("department", "").lower() == department.lower()
            ]

        actions = []
        for status in all_statuses:
            for course_status in status["course_statuses"]:
                if not course_status["compliant"]:
                    result = _post("/enroll", {
                        "employee_id": status["employee_id"],
                        "course_id": course_status["course_id"],
                    })
                    actions.append({
                        "employee": status["name"],
                        "employee_id": status["employee_id"],
                        "course": course_status["course_name"],
                        "course_id": course_status["course_id"],
                        "completion_pct": course_status["completion_pct"],
                        "enrollment_result": result.get("status", "error"),
                    })

        result = {
            "enrollments_made": len(actions),
            "department_filter": department or "all",
            "actions": actions,
        }
        print(f"[Compliance] ✓ Enrolled {len(actions)} employee-course pairs")
        return result

    def issue_certificates_for_compliant(self) -> dict:
        """
        Issue certificates for employees who completed mandatory courses but have no cert yet.
        """
        print("[Compliance] ▶ Issuing certificates for compliant employees...")
        all_statuses = _get("/compliance/status") or []
        existing_certs = _get("/certificates") or []
        certified_pairs = {(c["employee_id"], c["course_id"]) for c in existing_certs}

        issued = []
        for status in all_statuses:
            for cs in status["course_statuses"]:
                if cs["compliant"] and (status["employee_id"], cs["course_id"]) not in certified_pairs:
                    cert = _post("/certificates/issue", {
                        "employee_id": status["employee_id"],
                        "employee_name": status["name"],
                        "course_id": cs["course_id"],
                        "course_name": cs["course_name"],
                    })
                    if "id" in cert:
                        issued.append(cert)
                        certified_pairs.add((status["employee_id"], cs["course_id"]))

        result = {"certificates_issued": len(issued), "certificates": issued}
        print(f"[Compliance] ✓ Issued {len(issued)} certificates")
        return result

    # ── Reports ──────────────────────────────────────────────────────────────

    def generate_audit_report(self, compliance_data: dict) -> str:
        """Use Groq LLM to generate a formal compliance audit report."""
        print("[Compliance] ▶ Generating audit report via LLM...")
        prompt = f"""You are a compliance officer writing a formal training compliance audit report.

Write a structured audit report (4-5 paragraphs + summary table) covering:
1. Overall compliance rate and headline findings
2. Department-by-department breakdown
3. Non-compliant employees and their outstanding mandatory courses
4. Risk assessment (which gaps pose highest regulatory risk)
5. Recommended remediation actions with deadlines

Data:
{json.dumps(compliance_data, indent=2)}

Format: professional, clear, suitable for HR leadership. Use markdown headers and bullet points."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
        report = response.choices[0].message.content or ""
        print(f"[Compliance] ✓ Audit report generated ({len(report)} chars)")
        return report

    def generate_employee_notice(self, status: dict) -> str:
        """Generate a compliance notice for a non-compliant employee."""
        print(f"[Compliance] ▶ Generating compliance notice for {status.get('name', 'employee')}...")
        non_compliant_courses = [
            cs for cs in status["course_statuses"] if not cs["compliant"]
        ]
        prompt = f"""Write a professional but friendly compliance training notice to an employee.

Include:
- Which mandatory courses they still need to complete
- The deadline and consequences of non-completion (be firm but supportive)
- Encouragement and offer of support

Employee: {status['name']} ({status['role']})
Outstanding courses: {json.dumps(non_compliant_courses, indent=2)}

Keep it under 200 words. Professional tone."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        notice = response.choices[0].message.content or ""
        print(f"[Compliance] ✓ Notice generated for {status.get('name', 'employee')} ({len(notice)} chars)")
        return notice

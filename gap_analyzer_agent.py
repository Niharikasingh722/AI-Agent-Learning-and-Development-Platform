"""
Skill Gap Analyzer Agent
Compares employee current skills against the competency framework for their role/seniority.
Returns structured gap reports per employee and team-level rollup.
Uses Groq LLM to generate a human-readable narrative summary.
"""
import json
import os
from pathlib import Path
from groq import Groq

DATA_DIR = Path(__file__).parent.parent / "data"


def _load(filename: str):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def _compute_gap(employee: dict, framework: dict) -> dict:
    """Compute missing and present skills for one employee."""
    role = employee["role"]
    seniority = employee["seniority"]
    required = set(framework.get(role, {}).get(seniority, []))
    current = set(employee["current_skills"])
    missing = sorted(required - current)
    present = sorted(required & current)
    gap_pct = round(len(missing) / len(required) * 100) if required else 0
    return {
        "employee_id": employee["id"],
        "name": employee["name"],
        "role": role,
        "seniority": seniority,
        "required_skills": sorted(required),
        "current_skills": sorted(current),
        "missing_skills": missing,
        "present_skills": present,
        "gap_percentage": gap_pct,
        "skills_missing_count": len(missing),
        "skills_total_required": len(required),
    }


class SkillGapAnalyzerAgent:
    """
    Analyzes skill gaps for individual employees or entire departments.
    Returns structured data + LLM-generated narrative.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.employees = _load("employees.json")
        self.framework = _load("competency_framework.json")

    def analyze_employee(self, employee_id: str) -> dict:
        """Return gap analysis for a single employee."""
        emp = next((e for e in self.employees if e["id"] == employee_id), None)
        if not emp:
            print(f"[GapAnalyzer] ✗ Employee '{employee_id}' not found")
            return {"error": f"Employee {employee_id} not found"}
        print(f"[GapAnalyzer] ▶ Analyzing skill gaps for {emp['name']} ({emp['role']}, {emp['seniority']})...")
        result = _compute_gap(emp, self.framework)
        print(f"[GapAnalyzer] ✓ {emp['name']}: {result['skills_missing_count']}/{result['skills_total_required']} skills missing ({result['gap_percentage']}% gap)")
        if result['missing_skills']:
            print(f"[GapAnalyzer]   Missing: {', '.join(result['missing_skills'])}")
        return result

    def analyze_department(self, department: str) -> dict:
        """Return gap analysis for all employees in a department."""
        dept_employees = [e for e in self.employees if e["department"].lower() == department.lower()]
        if not dept_employees:
            print(f"[GapAnalyzer] ✗ No employees found in department '{department}'")
            return {"error": f"No employees found in department '{department}'"}

        print(f"[GapAnalyzer] ▶ Analyzing skill gaps for {department} department ({len(dept_employees)} employees)...")
        gaps = [_compute_gap(e, self.framework) for e in dept_employees]
        avg_gap = round(sum(g["gap_percentage"] for g in gaps) / len(gaps))

        # Count how often each skill is missing across the team
        skill_frequency: dict[str, int] = {}
        for g in gaps:
            for skill in g["missing_skills"]:
                skill_frequency[skill] = skill_frequency.get(skill, 0) + 1

        top_gaps = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)

        print(f"[GapAnalyzer] ✓ {department} dept average gap: {avg_gap}%")
        if top_gaps:
            print(f"[GapAnalyzer]   Top missing skills: {', '.join(s for s, _ in top_gaps[:3])}")

        return {
            "department": department,
            "employee_count": len(gaps),
            "average_gap_percentage": avg_gap,
            "top_missing_skills": top_gaps,
            "individual_gaps": gaps,
        }

    def analyze_all(self) -> dict:
        """Return gap analysis for every employee."""
        print(f"[GapAnalyzer] ▶ Analyzing skill gaps for all {len(self.employees)} employees...")
        gaps = [_compute_gap(e, self.framework) for e in self.employees]
        departments = {}
        for g in gaps:
            dept = next(e["department"] for e in self.employees if e["id"] == g["employee_id"])
            departments.setdefault(dept, []).append(g)

        dept_summaries = {}
        for dept, dept_gaps in departments.items():
            avg = round(sum(x["gap_percentage"] for x in dept_gaps) / len(dept_gaps))
            freq: dict[str, int] = {}
            for dg in dept_gaps:
                for sk in dg["missing_skills"]:
                    freq[sk] = freq.get(sk, 0) + 1
            dept_summaries[dept] = {
                "avg_gap_pct": avg,
                "top_missing_skills": sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5],
            }

        overall_avg = round(sum(g["gap_percentage"] for g in gaps) / len(gaps)) if gaps else 0
        print(f"[GapAnalyzer] ✓ All-org analysis complete: {len(gaps)} employees, {overall_avg}% average gap")
        return {
            "total_employees": len(gaps),
            "department_summaries": dept_summaries,
            "all_gaps": gaps,
        }

    def generate_narrative(self, gap_data: dict) -> str:
        """Use Groq LLM to generate a human-readable gap analysis summary."""
        print("[GapAnalyzer] ▶ Generating narrative summary via LLM...")
        prompt = f"""You are an L&D analyst. Based on the following skill gap analysis data, 
write a clear, actionable summary (3-5 paragraphs) for an HR manager.

Highlight:
- The most critical skill gaps
- Which teams or individuals need the most urgent attention
- Top 3 recommended training priorities

Data:
{json.dumps(gap_data, indent=2)}

Write in a professional but concise tone. Use bullet points where appropriate."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        narrative = response.choices[0].message.content or ""
        print(f"[GapAnalyzer] ✓ Narrative generated ({len(narrative)} chars)")
        return narrative

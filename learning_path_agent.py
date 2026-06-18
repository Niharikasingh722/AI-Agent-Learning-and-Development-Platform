"""
Learning Path Generator Agent
Takes an employee's gap analysis and queries the mock LMS to build
a personalized, sequenced learning plan with timeline.
Uses Groq LLM to generate a narrative plan and rationale.
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


def _get_lms_courses() -> list[dict]:
    """Fetch full course catalog from mock LMS."""
    try:
        resp = requests.get(f"{LMS_BASE}/courses", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Fallback: load directly from file if LMS not running
        return _load("courses.json")


def _match_courses_to_gaps(missing_skills: list[str], all_courses: list[dict]) -> list[dict]:
    """Find courses that address at least one missing skill."""
    matched = []
    covered_skills = set()
    for course in all_courses:
        overlap = [s for s in course["skills"] if s in missing_skills]
        if overlap:
            matched.append({**course, "addresses": overlap})
            covered_skills.update(overlap)

    # Sort: most skills addressed first, then shortest duration
    matched.sort(key=lambda c: (-len(c["addresses"]), c["duration_weeks"]))
    return matched


def _build_schedule(courses: list[dict]) -> list[dict]:
    """
    Build a week-by-week schedule.
    Assume one course at a time, sequential.
    """
    schedule = []
    week = 1
    for course in courses:
        schedule.append({
            "course_id": course["id"],
            "title": course["title"],
            "start_week": week,
            "end_week": week + course["duration_weeks"] - 1,
            "duration_weeks": course["duration_weeks"],
            "skills_covered": course["addresses"],
            "level": course["level"],
        })
        week += course["duration_weeks"]
    return schedule


class LearningPathGeneratorAgent:
    """
    Generates a personalized learning path for an employee based on their skill gaps.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.employees = _load("employees.json")
        self.framework = _load("competency_framework.json")

    def generate_path(self, employee_id: str, gap_data: dict | None = None) -> dict:
        """
        Build a learning path for the given employee.
        Accepts pre-computed gap_data or computes it internally.
        """
        emp = next((e for e in self.employees if e["id"] == employee_id), None)
        if not emp:
            print(f"[LearningPath] ✗ Employee '{employee_id}' not found")
            return {"error": f"Employee {employee_id} not found"}

        print(f"[LearningPath] ▶ Generating learning path for {emp['name']} ({emp['role']}, {emp['seniority']})...")

        if gap_data is None:
            from agents.gap_analyzer_agent import SkillGapAnalyzerAgent
            print(f"[LearningPath]   No gap data provided — running gap analysis for {emp['name']}")
            gap_data = SkillGapAnalyzerAgent().analyze_employee(employee_id)

        missing = gap_data.get("missing_skills", [])
        if not missing:
            print(f"[LearningPath] ✓ {emp['name']} has no skill gaps — fully compliant with role requirements")
            return {
                "employee_id": employee_id,
                "name": emp["name"],
                "message": "No skill gaps found — this employee meets all requirements for their role.",
                "schedule": [],
                "total_weeks": 0,
            }

        all_courses = _get_lms_courses()
        print(f"[LearningPath]   Matching {len(missing)} missing skills against {len(all_courses)} available courses...")
        matched = _match_courses_to_gaps(missing, all_courses)
        schedule = _build_schedule(matched)

        # Identify uncovered skills
        covered = {s for item in schedule for s in item["skills_covered"]}
        uncovered = [s for s in missing if s not in covered]

        total_weeks = schedule[-1]["end_week"] if schedule else 0

        print(f"[LearningPath] ✓ Learning path for {emp['name']}: {len(schedule)} courses over {total_weeks} weeks")
        if uncovered:
            print(f"[LearningPath]   Uncovered skills (no course available): {', '.join(uncovered)}")

        return {
            "employee_id": employee_id,
            "name": emp["name"],
            "role": emp["role"],
            "seniority": emp["seniority"],
            "missing_skills": missing,
            "covered_by_courses": sorted(covered),
            "not_covered_by_courses": uncovered,
            "recommended_courses": matched,
            "schedule": schedule,
            "total_weeks": total_weeks,
            "total_courses": len(schedule),
        }

    def generate_narrative(self, path_data: dict) -> str:
        """Use Groq LLM to write a personalised learning plan explanation."""
        print(f"[LearningPath] ▶ Generating personalised narrative for {path_data.get('name', 'employee')}...")
        prompt = f"""You are an L&D specialist creating a personalised development plan.

Write a motivating, clear learning plan narrative for the following employee (4-6 paragraphs):
- Address them by first name
- Explain what skills they will gain and why each matters for their role
- Walk through the course sequence and approximate timeline
- End with an encouraging call to action

Learning Path Data:
{json.dumps(path_data, indent=2)}

Tone: professional, supportive, forward-looking."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=900,
        )
        narrative = response.choices[0].message.content or ""
        print(f"[LearningPath] ✓ Narrative generated ({len(narrative)} chars)")
        return narrative

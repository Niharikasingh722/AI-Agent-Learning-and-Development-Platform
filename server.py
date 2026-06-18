"""
Mock LMS API — FastAPI stub server
Endpoints mirror a real LMS: course catalog, enrollment, completion, compliance, certificates
"""
import json
import uuid
from datetime import date, timedelta
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock LMS API", version="1.0")

DATA_DIR = Path(__file__).parent.parent / "data"

def load_json(filename: str):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)

# In-memory enrollment store  {employee_id: [course_id, ...]}
_enrollments: dict[str, list[str]] = {}
# In-memory completion store  {employee_id: {course_id: pct}}
_completions: dict[str, dict[str, int]] = {
    "E003": {"C002": 100, "C006": 75},
    "E005": {"C005": 100, "C003": 50},
    "E007": {"C003": 100, "C008": 100},
    "E009": {"C011": 100},
}

# In-memory retention log  {employee_id: [{course_id, completed_date}]}
_retention_log: dict[str, list[dict]] = {
    "E003": [{"course_id": "C002", "completed_date": "2025-11-15"}],
    "E005": [{"course_id": "C005", "completed_date": "2025-09-01"}],
    "E007": [
        {"course_id": "C003", "completed_date": "2025-07-10"},
        {"course_id": "C008", "completed_date": "2025-08-20"},
    ],
    "E009": [{"course_id": "C011", "completed_date": "2025-06-10"}],
}

# In-memory certificates (loaded from file + runtime-issued)
_certificates: list[dict] = []


# ── Courses ────────────────────────────────────────────────────────────────

@app.get("/courses")
def list_courses():
    """Return full course catalog."""
    return load_json("courses.json")


@app.get("/courses/{course_id}")
def get_course(course_id: str):
    """Return a single course by ID."""
    courses = load_json("courses.json")
    for c in courses:
        if c["id"] == course_id:
            return c
    raise HTTPException(status_code=404, detail=f"Course {course_id} not found")


@app.get("/courses/by-skill/{skill}")
def courses_by_skill(skill: str):
    """Return courses that teach a given skill."""
    courses = load_json("courses.json")
    return [c for c in courses if skill.lower() in [s.lower() for s in c["skills"]]]


# ── Employees ──────────────────────────────────────────────────────────────

@app.get("/employees")
def list_employees():
    return load_json("employees.json")


@app.get("/employees/{employee_id}")
def get_employee(employee_id: str):
    employees = load_json("employees.json")
    for e in employees:
        if e["id"] == employee_id:
            return e
    raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")


# ── Enrollment ─────────────────────────────────────────────────────────────

class EnrollRequest(BaseModel):
    employee_id: str
    course_id: str


@app.post("/enroll")
def enroll(req: EnrollRequest):
    """Enroll an employee in a course."""
    enrolled = _enrollments.setdefault(req.employee_id, [])
    if req.course_id not in enrolled:
        enrolled.append(req.course_id)
    return {"status": "enrolled", "employee_id": req.employee_id, "course_id": req.course_id}


@app.get("/enrollment/{employee_id}")
def get_enrollment(employee_id: str):
    """Return all courses an employee is enrolled in."""
    return {"employee_id": employee_id, "enrolled_courses": _enrollments.get(employee_id, [])}


# ── Completion ─────────────────────────────────────────────────────────────

@app.get("/completion/{employee_id}")
def get_completion(employee_id: str):
    """Return course completion records for an employee."""
    return {
        "employee_id": employee_id,
        "completions": _completions.get(employee_id, {})
    }


@app.get("/completion/{employee_id}/{course_id}")
def get_course_completion(employee_id: str, course_id: str):
    """Return completion percentage for a specific employee + course."""
    pct = _completions.get(employee_id, {}).get(course_id, 0)
    return {"employee_id": employee_id, "course_id": course_id, "completion_pct": pct}


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Mock LMS API"}


# ── Retention ──────────────────────────────────────────────────────────────

@app.get("/retention/{employee_id}")
def get_retention_log(employee_id: str):
    """Return completion history with dates (used for spaced repetition)."""
    return {
        "employee_id": employee_id,
        "history": _retention_log.get(employee_id, []),
    }


@app.get("/retention")
def get_all_retention():
    """Return full retention log for all employees."""
    return _retention_log


# ── Compliance ─────────────────────────────────────────────────────────────

@app.get("/compliance/rules")
def get_compliance_rules():
    """Return mandatory course rules per role."""
    return load_json("compliance_rules.json")


@app.get("/compliance/status/{employee_id}")
def get_compliance_status(employee_id: str):
    """Return compliance status for a single employee."""
    employees = load_json("employees.json")
    emp = next((e for e in employees if e["id"] == employee_id), None)
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    rules = load_json("compliance_rules.json")
    required_courses = rules["mandatory_courses"].get(emp["role"], [])
    deadlines = rules["deadlines_days"]
    course_names = rules["course_names"]
    completions = _completions.get(employee_id, {})

    statuses = []
    for cid in required_courses:
        pct = completions.get(cid, 0)
        statuses.append({
            "course_id": cid,
            "course_name": course_names.get(cid, cid),
            "completion_pct": pct,
            "compliant": pct == 100,
            "deadline_days": deadlines.get(cid, 90),
        })

    compliant_count = sum(1 for s in statuses if s["compliant"])
    return {
        "employee_id": employee_id,
        "name": emp["name"],
        "role": emp["role"],
        "department": emp["department"],
        "required_courses": len(statuses),
        "compliant_courses": compliant_count,
        "fully_compliant": compliant_count == len(statuses),
        "course_statuses": statuses,
    }


@app.get("/compliance/status")
def get_all_compliance():
    """Return compliance status for every employee."""
    employees = load_json("employees.json")
    return [get_compliance_status(e["id"]) for e in employees]


# ── Certificates ───────────────────────────────────────────────────────────

@app.get("/certificates")
def list_certificates():
    """Return all issued certificates (seed + runtime)."""
    seed = load_json("certificates.json")
    # Merge seed with any runtime-issued certs (avoid duplicates by id)
    seed_ids = {c["id"] for c in seed}
    runtime = [c for c in _certificates if c["id"] not in seed_ids]
    return seed + runtime


@app.get("/certificates/{employee_id}")
def get_employee_certificates(employee_id: str):
    """Return all certificates for a specific employee."""
    all_certs = list_certificates()
    return [c for c in all_certs if c["employee_id"] == employee_id]


class IssueCertRequest(BaseModel):
    employee_id: str
    employee_name: str
    course_id: str
    course_name: str


@app.post("/certificates/issue")
def issue_certificate(req: IssueCertRequest):
    """Issue a new certificate for an employee who completed a mandatory course."""
    today = date.today()
    cert = {
        "id": f"CERT-{uuid.uuid4().hex[:6].upper()}",
        "employee_id": req.employee_id,
        "employee_name": req.employee_name,
        "course_id": req.course_id,
        "course_name": req.course_name,
        "issued_date": today.isoformat(),
        "expires_date": (today + timedelta(days=365)).isoformat(),
        "status": "valid",
    }
    _certificates.append(cert)
    return cert

"""
Training Coach Agent
Answers learner questions grounded in course documents (RAG-style).
Uses Groq LLM (llama-3.3-70b-versatile).
"""
import os
from pathlib import Path
from groq import Groq

COURSE_DOCS_DIR = Path(__file__).parent.parent / "course_docs"

SYSTEM_PROMPT = """You are an expert Learning & Development coach embedded inside a corporate training platform.

Your job is to help employees understand course material, answer questions clearly, and guide their learning journey.

Guidelines:
- Answer ONLY from the course content provided in the context below.
- If the question is outside the provided material, say so honestly and suggest the learner contact their L&D team.
- Use concrete examples, analogies, and structured explanations.
- Keep answers focused and practical — learners are busy professionals.
- Cite the relevant module name when referencing specific content.
- Be encouraging and supportive in tone.

Course Content:
{course_context}
"""


def _load_course_docs() -> dict[str, str]:
    """Load all markdown course documents into a dict keyed by filename stem."""
    docs = {}
    for md_file in COURSE_DOCS_DIR.glob("*.md"):
        docs[md_file.stem] = md_file.read_text(encoding="utf-8")
    return docs


def _build_context(selected_course: str | None, docs: dict[str, str]) -> str:
    """Return course content as context string."""
    if selected_course and selected_course in docs:
        return f"=== {selected_course.replace('_', ' ').title()} ===\n\n{docs[selected_course]}"
    # No specific course selected — include all docs
    parts = []
    for name, content in docs.items():
        parts.append(f"=== {name.replace('_', ' ').title()} ===\n\n{content}")
    return "\n\n---\n\n".join(parts)


class TrainingCoachAgent:
    """
    Stateful conversational agent that answers learner questions
    based on course document content.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.docs = _load_course_docs()
        self.history: list[dict] = []
        self.selected_course: str | None = None

    def available_courses(self) -> list[str]:
        return list(self.docs.keys())

    def select_course(self, course_stem: str) -> str:
        """Select a course to focus coaching on."""
        if course_stem in self.docs:
            self.selected_course = course_stem
            self.history = []  # reset history on course switch
            print(f"[Coach] ▶ Course context set to: {course_stem.replace('_', ' ').title()}")
            return f"Coaching context set to: **{course_stem.replace('_', ' ').title()}**"
        print(f"[Coach] ✗ Course '{course_stem}' not found. Available: {self.available_courses()}")
        return f"Course '{course_stem}' not found. Available: {self.available_courses()}"

    def chat(self, user_message: str) -> str:
        """Send a message and get a coaching response."""
        course_label = self.selected_course.replace('_', ' ').title() if self.selected_course else "all courses"
        print(f"[Coach] ▶ Answering question (context: {course_label}): {user_message[:80]}{'...' if len(user_message) > 80 else ''}")
        context = _build_context(self.selected_course, self.docs)
        system = SYSTEM_PROMPT.format(course_context=context)

        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system}] + self.history

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,  # type: ignore[arg-type]
            temperature=0.4,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        print(f"[Coach] ✓ Response ready ({len(reply)} chars)")
        return reply

    def reset(self):
        """Clear conversation history."""
        self.history = []
        self.selected_course = None

"""
Content Generator Agent
Takes a skill gap and generates a short microlearning module in real-time.
Uses Groq streaming API so content appears token-by-token — visually compelling in demos.

Output per module:
  - Title
  - Learning objective
  - 3 key concepts with explanations
  - 2 quiz questions with answers
  - Practical exercise
"""
import os
from groq import Groq

SYSTEM_PROMPT = """You are an expert instructional designer creating a microlearning module.

Given a skill and target role, produce a concise, high-quality microlearning module in the following
EXACT markdown format — no deviations:

# [Module Title]

**Learning Objective:** [One clear sentence stating what the learner will be able to do]

---

## 🔑 Key Concepts

### 1. [Concept Name]
[2-3 sentence explanation, practical and jargon-free]

### 2. [Concept Name]
[2-3 sentence explanation, practical and jargon-free]

### 3. [Concept Name]
[2-3 sentence explanation, practical and jargon-free]

---

## 🧠 Knowledge Check

**Question 1:** [Question text]
- A) [Option]
- B) [Option]
- C) [Option]
- D) [Option]

✅ **Answer:** [Correct option letter] — [Brief explanation]

**Question 2:** [Question text]
- A) [Option]
- B) [Option]
- C) [Option]
- D) [Option]

✅ **Answer:** [Correct option letter] — [Brief explanation]

---

## 🛠️ Practical Exercise
[One concrete 15-minute exercise the learner can do today to practise this skill]

---
*Estimated reading time: 5 minutes*
"""


class ContentGeneratorAgent:
    """
    Generates microlearning modules for specific skills.
    Supports both streaming (for live demo) and non-streaming modes.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def generate_stream(self, skill: str, role: str, seniority: str = "Mid"):
        """
        Stream a microlearning module token-by-token.
        Yields string chunks — use with st.write_stream() for live demo effect.
        """
        print(f"[ContentGen] ▶ Streaming microlearning module: '{skill}' for {seniority} {role}...")
        prompt = (
            f"Create a microlearning module for the skill: **{skill}**\n"
            f"Target audience: {seniority} {role}\n"
            f"Keep it practical, concise, and immediately applicable."
        )

        stream = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1200,
            stream=True,
        )

        token_count = 0
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                token_count += 1
                yield delta
        print(f"[ContentGen] ✓ Module streamed ({token_count} chunks)")

    def generate(self, skill: str, role: str, seniority: str = "Mid") -> str:
        """Non-streaming version — returns the full module as a string."""
        print(f"[ContentGen] ▶ Generating microlearning module: '{skill}' for {seniority} {role}...")
        result = "".join(self.generate_stream(skill, role, seniority))
        print(f"[ContentGen] ✓ Module complete ({len(result)} chars)")
        return result

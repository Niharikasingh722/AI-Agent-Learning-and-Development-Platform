"""
L&D Agentic Orchestrator
LangGraph-based agent loop using Groq tool calling.

The LLM autonomously decides which specialist agents to invoke and in what order,
observes each result, and chains further calls as needed before producing a final
synthesised answer.

Conversation history is persisted to SQLite via agents.memory.
"""
import json
import os
from typing import Any, TypedDict

from groq import Groq
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from agents import memory as mem

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_employee_gaps",
            "description": (
                "Analyze skill gaps for a single named employee. "
                "Returns missing skills, present skills, and gap percentage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_name": {
                        "type": "string",
                        "description": "Full or partial name, e.g. 'Alice Chen' or 'Bob'.",
                    }
                },
                "required": ["employee_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_department_gaps",
            "description": "Analyze skill gaps across an entire department (Engineering, Analytics, HR).",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Department name, e.g. 'Engineering'.",
                    }
                },
                "required": ["department"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_all_gaps",
            "description": "Analyze skill gaps for every employee across all departments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_learning_path",
            "description": (
                "Generate a personalised sequenced learning path for an employee based on their skill gaps. "
                "Call analyze_employee_gaps first if you also need to discuss specific gaps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_name": {
                        "type": "string",
                        "description": "Full or partial name of the employee.",
                    }
                },
                "required": ["employee_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_training_coach",
            "description": (
                "Ask the training coach a question about course content, learning concepts, "
                "or training material. Grounded in actual course documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_employee_retention",
            "description": "Check knowledge retention for a single employee using the Ebbinghaus forgetting curve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_name": {
                        "type": "string",
                        "description": "Full or partial name of the employee.",
                    }
                },
                "required": ["employee_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_team_retention",
            "description": "Get a company-wide knowledge retention health overview for all employees.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_compliance",
            "description": (
                "Check mandatory training compliance. "
                "Provide employee_name for one person, department for a team, "
                "or omit both for a company-wide audit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_name": {
                        "type": "string",
                        "description": "Employee name (optional).",
                    },
                    "department": {
                        "type": "string",
                        "description": "Department name (optional).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enroll_non_compliant",
            "description": "Auto-enroll all non-compliant employees in their missing mandatory courses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Limit to this department (optional — omit for company-wide).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_certificates",
            "description": "Issue training certificates for all employees who have completed mandatory courses.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_microlearning",
            "description": (
                "Generate a microlearning module for a specific skill. "
                "Useful after gap analysis to immediately produce targeted training content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "The skill to create a module for, e.g. 'Docker'.",
                    },
                    "role": {
                        "type": "string",
                        "description": "Target employee role, e.g. 'Software Engineer'.",
                    },
                    "seniority": {
                        "type": "string",
                        "description": "Seniority level.",
                        "enum": ["Junior", "Mid", "Senior"],
                    },
                },
                "required": ["skill", "role"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an intelligent Learning & Development AI agent for a corporate training platform.

You have access to specialist tools covering skill gap analysis, learning paths, compliance,
knowledge retention, and content generation.

Guidelines:
- ALWAYS use tools to retrieve real data — never fabricate employee names, gaps, or compliance figures.
- For multi-step tasks (e.g. "analyse gaps then build a plan"), call tools in logical order and chain results.
- After receiving tool results, synthesise them into a clear, actionable markdown response.
- If a task naturally chains (e.g. gaps feed into a learning path), proactively call both tools.
- Be concise but thorough. Reference specific data from tool results in your answer.
- Maintain context from earlier messages — refer back to previous results when relevant.
"""

MAX_ITERATIONS = 5


class OrchestratorState(TypedDict):
    """State carried between LangGraph nodes."""

    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    tools_used: list[str]
    answer: str
    llm_turns: int
    done: bool
    last_tool_event: dict[str, Any] | None


class AgenticOrchestrator:
    """
    LangGraph-based agentic orchestrator using Groq tool calling.

    The LLM selects and sequences tools autonomously through a graph with two
    nodes: LLM decision and tool execution. Results feed back into the graph
    loop until a final synthesised answer is produced.
    Conversation history persists to SQLite via agents.memory.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        mem.init_db()
        # Lazily initialised specialist agents
        self._gap = None
        self._path = None
        self._coach = None
        self._retention = None
        self._compliance = None
        self._content = None
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph workflow for LLM/tool reasoning."""
        builder = StateGraph(OrchestratorState)
        builder.add_node("llm_node", self._llm_node)
        builder.add_node("tool_node", self._tool_node)

        builder.add_edge(START, "llm_node")
        builder.add_conditional_edges(
            "llm_node",
            self._route_after_llm,
            {
                "tool_node": "tool_node",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "tool_node",
            self._route_after_tool,
            {
                "tool_node": "tool_node",
                "llm_node": "llm_node",
            },
        )
        return builder.compile()

    def _route_after_llm(self, state: OrchestratorState) -> str:
        """Route to tool execution or finish based on LLM decision."""
        return "end" if state["done"] else "tool_node"

    def _route_after_tool(self, state: OrchestratorState) -> str:
        """Keep executing tools until queue is empty, then return to LLM."""
        return "tool_node" if state["pending_tool_calls"] else "llm_node"

    def _llm_node(self, state: OrchestratorState) -> dict[str, Any]:
        """LLM reasoning step: either emit tool calls or produce final answer."""
        next_turn = state["llm_turns"] + 1
        if next_turn > MAX_ITERATIONS:
            summary_messages = state["messages"] + [
                {"role": "user", "content": "Please summarise what you found so far."}
            ]
            writer = get_stream_writer()
            stream = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=summary_messages,  # type: ignore[arg-type]
                temperature=0.1,
                max_tokens=2048,
                stream=True,
            )
            answer_parts: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    answer_parts.append(delta)
                    writer({"type": "answer_delta", "delta": delta})
            answer = "".join(answer_parts)
            print(
                f"\n[Orchestrator] ✓ Final answer ready ({len(answer)} chars, "
                f"tools used: {state['tools_used'] or 'none'})"
            )
            return {
                "answer": answer,
                "done": True,
                "llm_turns": next_turn,
                "last_tool_event": None,
            }

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=state["messages"],  # type: ignore[arg-type]
            tools=TOOLS,  # type: ignore[arg-type]
            tool_choice="auto",
            temperature=0.1,
            max_tokens=4096,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            writer = get_stream_writer()
            stream = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=state["messages"],  # type: ignore[arg-type]
                temperature=0.1,
                max_tokens=4096,
                stream=True,
            )
            answer_parts: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    answer_parts.append(delta)
                    writer({"type": "answer_delta", "delta": delta})
            answer = "".join(answer_parts)
            print(
                f"\n[Orchestrator] ✓ Final answer ready ({len(answer)} chars, "
                f"tools used: {state['tools_used'] or 'none'})"
            )
            return {
                "answer": answer,
                "done": True,
                "llm_turns": next_turn,
                "last_tool_event": None,
            }

        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }

        pending_tool_calls = []
        for tc in msg.tool_calls:
            pending_tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments or "{}"),
                }
            )

        return {
            "messages": state["messages"] + [assistant_msg],
            "pending_tool_calls": pending_tool_calls,
            "done": False,
            "llm_turns": next_turn,
            "last_tool_event": None,
        }

    def _tool_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Execute one tool call per step so UI can stream events in real time."""
        if not state["pending_tool_calls"]:
            return {"last_tool_event": None}

        messages = list(state["messages"])
        tool_trace = list(state["tool_trace"])
        tools_used = list(state["tools_used"])

        tc = state["pending_tool_calls"][0]
        remaining_tool_calls = state["pending_tool_calls"][1:]
        name = tc["name"]
        args = tc["args"]

        print(f"[Orchestrator]   LLM selected tool: {name}")
        result = self._execute_tool(name, args)
        print(f"[Orchestrator]   Result: {result[:200]}{'...' if len(result) > 200 else ''}")

        tool_event = {"tool": name, "args": args, "result": result}
        tool_trace.append(tool_event)
        tools_used.append(name)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            }
        )

        return {
            "messages": messages,
            "tool_trace": tool_trace,
            "tools_used": tools_used,
            "pending_tool_calls": remaining_tool_calls,
            "last_tool_event": tool_event,
        }

    # ── Agent accessors ───────────────────────────────────────────────────────

    @property
    def gap(self):
        if not self._gap:
            from agents.gap_analyzer_agent import SkillGapAnalyzerAgent
            self._gap = SkillGapAnalyzerAgent()
        return self._gap

    @property
    def path(self):
        if not self._path:
            from agents.learning_path_agent import LearningPathGeneratorAgent
            self._path = LearningPathGeneratorAgent()
        return self._path

    @property
    def coach(self):
        if not self._coach:
            from agents.coach_agent import TrainingCoachAgent
            self._coach = TrainingCoachAgent()
        return self._coach

    @property
    def retention(self):
        if not self._retention:
            from agents.retention_agent import KnowledgeRetentionAgent
            self._retention = KnowledgeRetentionAgent()
        return self._retention

    @property
    def compliance(self):
        if not self._compliance:
            from agents.compliance_agent import ComplianceAgent
            self._compliance = ComplianceAgent()
        return self._compliance

    @property
    def content(self):
        if not self._content:
            from agents.content_generator_agent import ContentGeneratorAgent
            self._content = ContentGeneratorAgent()
        return self._content

    # ── Employee lookup ───────────────────────────────────────────────────────

    def _find_employee(self, name: str) -> tuple[str | None, str | None]:
        """Return (employee_id, full_name) for the first match, or (None, None)."""
        if not name:
            return None, None
        name_lower = name.lower()
        for e in self.gap.employees:
            if name_lower in e["name"].lower():
                return e["id"], e["name"]
        return None, None

    # ── Tool executor ─────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> str:
        """Run a named tool and return its result as a JSON string."""
        print(f"\n[Orchestrator] ▶ Tool call: {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
        try:
            if name == "analyze_employee_gaps":
                emp_id, _ = self._find_employee(args.get("employee_name", ""))
                if not emp_id:
                    return json.dumps({"error": f"Employee '{args.get('employee_name')}' not found."})
                gap = self.gap.analyze_employee(emp_id)
                gap["narrative"] = self.gap.generate_narrative(gap)
                mem.cache_set(f"gap:{emp_id}", gap)  # cache for agent-to-agent handoff
                return json.dumps(gap)

            elif name == "analyze_department_gaps":
                data = self.gap.analyze_department(args.get("department", ""))
                if "error" not in data:
                    data["narrative"] = self.gap.generate_narrative(data)
                return json.dumps(data)

            elif name == "analyze_all_gaps":
                data = self.gap.analyze_all()
                data["narrative"] = self.gap.generate_narrative(data)
                return json.dumps(data)

            elif name == "generate_learning_path":
                emp_id, _ = self._find_employee(args.get("employee_name", ""))
                if not emp_id:
                    return json.dumps({"error": f"Employee '{args.get('employee_name')}' not found."})
                # Agent-to-agent handoff: reuse cached gap data if the gap tool was called earlier
                gap_data = mem.cache_get(f"gap:{emp_id}")
                path = self.path.generate_path(emp_id, gap_data=gap_data)
                if "error" not in path and path.get("schedule"):
                    path["narrative"] = self.path.generate_narrative(path)
                return json.dumps(path)

            elif name == "ask_training_coach":
                reply = self.coach.chat(args.get("question", ""))
                return json.dumps({"answer": reply})

            elif name == "check_employee_retention":
                emp_id, _ = self._find_employee(args.get("employee_name", ""))
                if not emp_id:
                    return json.dumps({"error": f"Employee '{args.get('employee_name')}' not found."})
                data = self.retention.analyze_employee(emp_id)
                if "course_scores" in data:
                    data["reminder"] = self.retention.generate_reminder(data)
                return json.dumps(data)

            elif name == "check_team_retention":
                data = self.retention.analyze_all()
                data["summary"] = self.retention.generate_team_summary(data)
                return json.dumps(data)

            elif name == "check_compliance":
                emp_name = args.get("employee_name")
                dept = args.get("department")
                if emp_name:
                    emp_id, _ = self._find_employee(emp_name)
                    if not emp_id:
                        return json.dumps({"error": f"Employee '{emp_name}' not found."})
                    data = self.compliance.check_employee(emp_id)
                    if not data.get("fully_compliant"):
                        data["notice"] = self.compliance.generate_employee_notice(data)
                elif dept:
                    data = self.compliance.check_department(dept)
                    if "error" not in data:
                        data["report"] = self.compliance.generate_audit_report(data)
                else:
                    data = self.compliance.check_all()
                    if "error" not in data:
                        data["report"] = self.compliance.generate_audit_report(data)
                return json.dumps(data)

            elif name == "enroll_non_compliant":
                result = self.compliance.enroll_non_compliant(args.get("department"))
                return json.dumps(result)

            elif name == "issue_certificates":
                result = self.compliance.issue_certificates_for_compliant()
                return json.dumps(result)

            elif name == "generate_microlearning":
                module_text = self.content.generate(
                    args.get("skill", ""),
                    args.get("role", ""),
                    args.get("seniority", "Mid"),
                )
                return json.dumps({"module": module_text})

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as exc:
            print(f"[Orchestrator] ✗ Tool '{name}' raised exception: {exc}")
            return json.dumps({"error": str(exc)})

    # ── LangGraph loop (UI-compatible event stream) ──────────────────────────

    def run(self, user_input: str, session_id: str):
        """
        Execute the LangGraph workflow and yield UI progress events.

        Yields dicts with one of these shapes:
          {"type": "tool_call",   "tool": str, "args": dict}
          {"type": "tool_result", "tool": str, "result": str}
          {"type": "answer",      "content": str, "tools_used": list[str]}
        """
        history = mem.load_history(session_id, limit=20)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_input})

        print(f"\n{'='*60}")
        print(f"[Orchestrator] ▶ New request (session: {session_id})")
        print(f"[Orchestrator]   User: {user_input[:120]}{'...' if len(user_input) > 120 else ''}")
        print(f"{'='*60}")

        initial_state: OrchestratorState = {
            "messages": messages,
            "pending_tool_calls": [],
            "tool_trace": [],
            "tools_used": [],
            "answer": "",
            "llm_turns": 0,
            "done": False,
            "last_tool_event": None,
        }

        state_snapshot: OrchestratorState = dict(initial_state)

        for update in self.graph.stream(
            initial_state,
            config={"recursion_limit": MAX_ITERATIONS * 8},
            stream_mode=["updates", "custom"],
        ):
            mode, payload = update
            if mode == "updates":
                for _, node_update in payload.items():
                    state_snapshot.update(node_update)

                    tool_event = node_update.get("last_tool_event")
                    if tool_event:
                        yield {
                            "type": "tool_call",
                            "tool": tool_event["tool"],
                            "args": tool_event["args"],
                        }
                        yield {
                            "type": "tool_result",
                            "tool": tool_event["tool"],
                            "result": tool_event["result"],
                        }
            elif mode == "custom":
                if payload.get("type") == "answer_delta" and payload.get("delta"):
                    yield {
                        "type": "answer_delta",
                        "delta": payload["delta"],
                    }

        tools_used = state_snapshot.get("tools_used", [])
        answer = state_snapshot.get("answer", "")
        if not answer:
            answer = "I could not generate a final response. Please try rephrasing your request."

        mem.save_message(session_id, "user", user_input)
        mem.save_message(session_id, "assistant", answer, tools_used=tools_used or None)
        yield {"type": "answer", "content": answer, "tools_used": tools_used}


# ── Legacy stub (kept for any direct imports) ─────────────────────────────────

class Orchestrator(AgenticOrchestrator):
    """Backwards-compatible alias."""

    def classify(self, user_input: str, history=None) -> dict:  # type: ignore[override]
        """Deprecated: use run() instead."""
        return {"intent": "UNKNOWN"}



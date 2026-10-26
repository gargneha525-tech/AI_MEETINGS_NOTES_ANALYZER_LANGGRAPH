# ============================================================
# PROJECT 8: AI MEETING NOTES ANALYZER
# File: workflow.py
# Technologies: LangGraph, LangChain, Ollama, Pydantic
# ============================================================

import json
import os
import re
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------

load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=TEMPERATURE,
)


# ------------------------------------------------------------
# 2. PYDANTIC OUTPUT SCHEMAS
# ------------------------------------------------------------

class TopicsResult(BaseModel):
    topics: list[str] = Field(
        default_factory=list,
        description="Main topics explicitly discussed in the meeting.",
    )


class SummaryResult(BaseModel):
    summary: str = Field(
        description="A concise, factual meeting summary."
    )


class ActionItem(BaseModel):
    task: str = Field(
        description="A clearly described action or agreed task."
    )
    owner: str = Field(
        default="Not specified",
        description=(
            "Person responsible for the task. "
            "Use 'Not specified' if the transcript does not identify one."
        ),
    )


class ActionItemsResult(BaseModel):
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Tasks assigned or agreed upon in the meeting.",
    )


class TaskPriority(BaseModel):
    task_id: int = Field(
        description="1-based index corresponding to the given action item."
    )
    task: str = Field(
        description="The task description being classified."
    )
    priority: Literal["High Priority", "Medium Priority", "Low Priority"] = Field(
        description="Priority level according to urgency and deadlines."
    )
    reason: str = Field(
        description="Evidence-based explanation from the transcript."
    )


class PrioritiesResult(BaseModel):
    priorities: list[TaskPriority] = Field(
        default_factory=list,
        description="Priority classification for each action item.",
    )


# ------------------------------------------------------------
# 3. LANGGRAPH STATE
# ------------------------------------------------------------

class MeetingState(TypedDict, total=False):
    transcript: str
    topics: list[str]
    summary: str
    action_items: list[dict[str, str]]
    priorities: list[dict[str, str]]
    final_report: dict[str, Any]
    error: Optional[str]


# ------------------------------------------------------------
# 4. INPUT VALIDATION NODE
# ------------------------------------------------------------

def input_node(state: MeetingState) -> MeetingState:
    """Validate and initialize the user-provided transcript."""

    transcript = state.get("transcript", "")

    if not isinstance(transcript, str):
        raise ValueError("The meeting transcript must be text.")

    transcript = transcript.strip()

    if not transcript:
        raise ValueError("Please provide a meeting transcript.")

    if len(transcript) < 10:
        raise ValueError(
            "The transcript is too short. "
            "Please provide more meeting details."
        )

    return {
        "transcript": transcript,
        "topics": [],
        "summary": "",
        "action_items": [],
        "priorities": [],
        "error": None,
    }


# ------------------------------------------------------------
# 5. TOPIC EXTRACTION AGENT
# ------------------------------------------------------------

def topic_extraction_agent(state: MeetingState) -> MeetingState:
    """Extract distinct discussion topics."""

    structured_llm = llm.with_structured_output(TopicsResult)

    prompt = f"""You are a professional meeting analyst.

Extract the main discussion topics from the transcript.

Rules:
- Use only information stated in the transcript.
- Keep topics concise and distinct.
- Remove duplicates.
- Do not invent topics.
- Return an empty list if no meaningful topics exist.

TRANSCRIPT:
{state["transcript"]}
"""

    result: TopicsResult = structured_llm.invoke(prompt)

    topics = []
    seen = set()

    raw_topics = getattr(result, "topics", []) or []
    for topic in raw_topics:
        topic_clean = topic.strip()
        key = topic_clean.casefold()

        if topic_clean and key not in seen:
            topics.append(topic_clean)
            seen.add(key)

    return {"topics": topics}


# ------------------------------------------------------------
# 6. MEETING SUMMARY AGENT
# ------------------------------------------------------------

def meeting_summary_agent(state: MeetingState) -> MeetingState:
    """Generate a concise and factual summary."""

    structured_llm = llm.with_structured_output(SummaryResult)

    prompt = f"""You are an expert meeting summarization assistant.

Summarize the meeting in approximately 3-5 concise sentences.

Include, when available:
- Meeting purpose.
- Main issues discussed.
- Decisions made.
- Important deadlines.
- Agreed next steps.

Rules:
- Do not invent information.
- Do not invent decisions or deadlines.
- Distinguish confirmed decisions from suggestions.
- Use only the supplied transcript.

TRANSCRIPT:
{state["transcript"]}
"""

    result: SummaryResult = structured_llm.invoke(prompt)
    summary_text = getattr(result, "summary", "").strip()

    return {"summary": summary_text}


# ------------------------------------------------------------
# 7. ACTION ITEM EXTRACTION AGENT
# ------------------------------------------------------------

def action_item_extraction_agent(state: MeetingState) -> MeetingState:
    """Identify tasks and the people responsible for them."""

    structured_llm = llm.with_structured_output(ActionItemsResult)

    prompt = f"""You are a meeting action-item extraction assistant.

Identify tasks explicitly assigned or agreed upon in the meeting.

For every action item, return:
- task: task description, including any stated deadline.
- owner: the responsible person.

Rules:
- Do not treat general discussion topics as tasks.
- Include tasks someone agrees to perform or the group agrees must be completed.
- Do not invent owners or deadlines.
- If an owner is not clearly stated, use "Not specified".
- Remove duplicate tasks.
- Return an empty list if no action items exist.
- Use only the supplied transcript.

TRANSCRIPT:
{state["transcript"]}
"""

    result: ActionItemsResult = structured_llm.invoke(prompt)

    action_items = []
    seen_tasks = set()

    raw_items = getattr(result, "action_items", []) or []
    for item in raw_items:
        task = item.task.strip()
        owner = item.owner.strip() or "Not specified"

        if not task:
            continue

        if owner.casefold() in {
            "unknown",
            "not mentioned",
            "not specified",
            "none",
            "n/a",
        }:
            owner = "Not specified"

        task_key = re.sub(r"\s+", " ", task).casefold()

        if task_key in seen_tasks:
            continue

        action_items.append({
            "task": task,
            "owner": owner,
        })
        seen_tasks.add(task_key)

    return {
        "action_items": action_items,
        "priorities": [],
    }


# ------------------------------------------------------------
# 8. CONDITIONAL ROUTING
# ------------------------------------------------------------

def route_after_action_extraction(
    state: MeetingState,
) -> Literal["classify_priority", "no_action_items"]:
    """Skip the priority agent when there are no tasks."""

    if state.get("action_items"):
        return "classify_priority"

    return "no_action_items"


# ------------------------------------------------------------
# 9. PRIORITY CLASSIFICATION AGENT
# ------------------------------------------------------------

def priority_classification_agent(state: MeetingState) -> MeetingState:
    """Assign a priority and explanation to every action item using structured output."""

    transcript = state["transcript"]
    action_items = state.get("action_items", [])

    if not action_items:
        return {"priorities": []}

    structured_llm = llm.with_structured_output(PrioritiesResult)

    task_list = "\n".join(
        f'{i + 1}. {item["task"]} (Owner: {item["owner"]})'
        for i, item in enumerate(action_items)
    )

    prompt = f"""You are a task priority classification assistant.

Classify every action item using evidence from the transcript.

TRANSCRIPT:
{transcript}

ACTION ITEMS TO CLASSIFY:
{task_list}

Classification Criteria:
- High Priority: Urgently required, critical blockers, or imminent deadlines (immediate / due by tomorrow).
- Medium Priority: Important operational tasks scheduled within the upcoming week (e.g., due in 2-7 days).
- Low Priority: Routine tasks, optional documentation, or tasks due well in the future (more than a week away / no pressing timeline).

Rules:
- Provide exactly one classification corresponding to each of the {len(action_items)} action items.
- task_id must match the 1-based number (1 to {len(action_items)}).
"""

    try:
        result: PrioritiesResult = structured_llm.invoke(prompt)
        raw_priorities = getattr(result, "priorities", []) or []
    except Exception:
        raw_priorities = []

    # Map responses by both task_id and normalized task text for robust fallback
    by_id: dict[int, TaskPriority] = {}
    by_name: dict[str, TaskPriority] = {}

    for item in raw_priorities:
        if item.task_id:
            by_id[item.task_id] = item
        if item.task:
            cleaned_key = re.sub(r"\s+", " ", item.task).casefold()
            by_name[cleaned_key] = item

    aligned_priorities = []

    for idx, action in enumerate(action_items, start=1):
        task_text = action["task"]
        task_key = re.sub(r"\s+", " ", task_text).casefold()

        matched = by_id.get(idx) or by_name.get(task_key)

        if matched and matched.priority:
            priority = matched.priority
            reason = matched.reason or "Priority determined from the meeting context."
        else:
            priority = "Medium Priority"
            reason = (
                "The transcript does not establish a specific urgency. "
                "Defaulted to Medium Priority."
            )

        aligned_priorities.append({
            "task": task_text,
            "priority": priority,
            "reason": reason,
        })

    return {"priorities": aligned_priorities}


# ------------------------------------------------------------
# 10. FINAL REPORT NODE
# ------------------------------------------------------------

def final_output_node(state: MeetingState) -> MeetingState:
    """Combine agent outputs into the final report."""

    topics = state.get("topics", [])
    summary = state.get("summary", "")
    action_items = state.get("action_items", [])
    priorities = state.get("priorities", [])

    priority_lookup = {
        re.sub(r"\s+", " ", item["task"]).casefold(): item
        for item in priorities
        if "task" in item
    }

    final_action_items = []

    for idx, action in enumerate(action_items):
        task = action["task"]
        task_key = re.sub(r"\s+", " ", task).casefold()

        priority_data = priority_lookup.get(task_key)
        if not priority_data and idx < len(priorities):
            priority_data = priorities[idx]

        if priority_data:
            priority = priority_data.get("priority", "Medium Priority")
            reason = priority_data.get(
                "reason", "Assigned based on meeting context."
            )
        else:
            priority = "Medium Priority"
            reason = "Urgency not specified; defaulted to Medium Priority."

        final_action_items.append({
            "task": task,
            "owner": action.get("owner", "Not specified"),
            "priority": priority,
            "priority_reason": reason,
        })

    if not action_items:
        priority_status = (
            "No action items identified. Priority classification skipped."
        )
    else:
        priority_status = "Priority classification completed."

    final_report = {
        "meeting_summary": summary,
        "key_topics": topics,
        "action_items": final_action_items,
        "task_owners": [
            {
                "task": item["task"],
                "owner": item["owner"],
            }
            for item in final_action_items
        ],
        "priority_status": priority_status,
        "total_topics": len(topics),
        "total_action_items": len(final_action_items),
    }

    return {"final_report": final_report}


# ------------------------------------------------------------
# 11. BUILD LANGGRAPH WORKFLOW
# ------------------------------------------------------------

workflow_builder = StateGraph(MeetingState)

workflow_builder.add_node("input", input_node)
workflow_builder.add_node("topic_extraction", topic_extraction_agent)
workflow_builder.add_node("meeting_summary", meeting_summary_agent)
workflow_builder.add_node(
    "action_item_extraction",
    action_item_extraction_agent,
)
workflow_builder.add_node(
    "priority_classification",
    priority_classification_agent,
)
workflow_builder.add_node("final_output", final_output_node)

workflow_builder.add_edge(START, "input")
workflow_builder.add_edge("input", "topic_extraction")
workflow_builder.add_edge("topic_extraction", "meeting_summary")
workflow_builder.add_edge(
    "meeting_summary",
    "action_item_extraction",
)

workflow_builder.add_conditional_edges(
    "action_item_extraction",
    route_after_action_extraction,
    {
        "classify_priority": "priority_classification",
        "no_action_items": "final_output",
    },
)

workflow_builder.add_edge("priority_classification", "final_output")
workflow_builder.add_edge("final_output", END)

meeting_graph = workflow_builder.compile()


# ------------------------------------------------------------
# 12. PUBLIC ANALYSIS FUNCTION
# ------------------------------------------------------------

def analyze_meeting(transcript: str) -> dict:
    """Analyze a user-supplied transcript and return a report."""

    if not isinstance(transcript, str):
        raise ValueError("The transcript must be text.")

    transcript = transcript.strip()

    if not transcript:
        raise ValueError("Please provide a meeting transcript.")

    result = meeting_graph.invoke({"transcript": transcript})

    if "final_report" not in result:
        raise RuntimeError(
            "The LangGraph workflow did not generate a final report."
        )

    return result["final_report"]


# ------------------------------------------------------------
# 13. COMMAND-LINE EXECUTION INTERFACE
# ------------------------------------------------------------

if __name__ == "__main__":
    print("\nAI Meeting Notes Analyzer")
    print("Enter your meeting transcript.")
    print("Type END on a separate line when finished.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip().upper() == "END":
            break

        lines.append(line)

    user_transcript = "\n".join(lines).strip()

    if not user_transcript:
        print("No transcript was entered. Nothing to analyze.")
    else:
        try:
            result = analyze_meeting(user_transcript)
            print("\n" + "=" * 60)
            print("MEETING ANALYSIS REPORT")
            print("=" * 60)

            print("\n1. MEETING SUMMARY:\n")
            print(result.get("meeting_summary", "No summary generated."))

            print("\n2. KEY DISCUSSION TOPICS:")
            topics = result.get("key_topics", [])
            if topics:
                for idx, topic in enumerate(topics, 1):
                    print(f"  {idx}. {topic}")
            else:
                print("  No topics identified.")

            print("\n3. ACTION ITEMS & PRIORITY CLASSIFICATION:")
            action_items = result.get("action_items", [])
            if action_items:
                for idx, item in enumerate(action_items, 1):
                    print(f"  [{idx}] Task: {item['task']}")
                    print(f"      Owner: {item['owner']} | Priority: {item['priority']}")
                    print(f"      Reason: {item['priority_reason']}\n")
            else:
                print("  No action items identified.")

            print("4. SUMMARY METRICS:")
            print(f"  Status: {result.get('priority_status', 'N/A')}")
            print(f"  Total Topics: {result.get('total_topics', 0)}")
            print(f"  Total Action Items: {result.get('total_action_items', 0)}")
            print("=" * 60 + "\n")
        except Exception as exc:
            print(f"Analysis failed: {exc}")
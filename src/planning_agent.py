import json
import re
import ast
from typing import List
from datetime import datetime

from src.model_manager import get_planner_model
from src.agents import research_agent, writer_agent, editor_agent


def planner_agent(topic: str) -> List[str]:
    print("==================================")
    print("Planner Agent (Mistral-7B)")
    print("==================================")
    
    prompt = f"""
You are a planning agent organizing a research workflow.

**Available agents:**
- Research agent: Uses DuckDuckGo (web), arXiv (academic), Wikipedia (encyclopedia)
- Writer agent: Drafts reports from research
- Editor agent: Reviews and improves drafts

**Task:** Create a 7-step research plan as a valid Python list of strings.

**Requirements:**
1. Each step is one string assigned to one agent
2. Step 1 MUST be: "Research agent: Use DuckDuckGo to perform broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
3. Step 2 MUST be: "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."
4. Final step MUST include: "Writer agent: Generate comprehensive Markdown report with inline citations and complete References section with clickable links."
5. NO steps about CSV creation, repo setup, or package installation
6. Focus on research tasks: search, extract, rank, draft, revise

**Output format:** Valid Python list only, no markdown, no explanations.

**Topic:** "{topic}"

**Example output:**
[
    "Research agent: Use DuckDuckGo to perform broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available).",
    "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist).",
    "Research agent: Synthesize findings by relevance and authority; deduplicate by title/DOI.",
    "Writer agent: Draft structured outline based on ranked evidence.",
    "Editor agent: Review for coherence and citation completeness.",
    "Writer agent: Generate comprehensive Markdown report with inline citations and complete References section with clickable links."
]
"""
    
    model = get_planner_model()
    
    raw = model.chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7
    )

    print(f"Raw planner output: {raw[:300]}...")

    # Parse list from response
    def _coerce_to_list(s: str) -> List[str]:
        # Remove markdown fences
        s = re.sub(r'```(?:python|json)?\n?', '', s)
        s = re.sub(r'\n?```', '', s)
        s = s.strip()
        
        # Try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except json.JSONDecodeError:
            pass
        
        # Try Python literal
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except Exception:
            pass
        
        return []

    steps = _coerce_to_list(raw)

    # Enforce contract
    required_first = "Research agent: Use DuckDuckGo to perform broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
    required_second = "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."
    final_required = "Writer agent: Generate the final comprehensive Markdown report with inline citations and a complete References section with clickable links."

    def _ensure_contract(steps_list: List[str]) -> List[str]:
        if not steps_list:
            return [
                required_first,
                required_second,
                "Research agent: Synthesize and rank findings by relevance, recency, and authority; deduplicate by title/DOI.",
                "Writer agent: Draft a structured outline based on the ranked evidence.",
                "Editor agent: Review for coherence, coverage, and citation completeness; request fixes.",
                final_required,
            ]
        
        # Inject/replace first two if missing or out of order
        steps_list = [s for s in steps_list if isinstance(s, str)]
        if not steps_list or steps_list[0] != required_first:
            steps_list = [required_first] + steps_list
        if len(steps_list) < 2 or steps_list[1] != required_second:
            steps_list = (
                [steps_list[0]]
                + [required_second]
                + [
                    s
                    for s in steps_list[1:]
                    if "arXiv" not in s or "For each collected item" in s
                ]
            )
        
        # Ensure final step requirement present
        if final_required not in steps_list:
            steps_list.append(final_required)
        
        # Cap to 7
        return steps_list[:7]

    steps = _ensure_contract(steps)

    return steps


def executor_agent_step(step_title: str, history: list, prompt: str):
    """
    Executes a step of the executor agent.
    Returns:
        - step_title (str)
        - agent_name (str)
        - output (str)
    """

    # Build enriched context from history
    context = f"User Prompt:\n{prompt}\n\nHistory so far:\n"
    for i, (desc, agent, output) in enumerate(history):
        if "draft" in desc.lower() or agent == "writer_agent":
            context += f"\nDraft (Step {i + 1}):\n{output.strip()}\n"
        elif "feedback" in desc.lower() or agent == "editor_agent":
            context += f"\nFeedback (Step {i + 1}):\n{output.strip()}\n"
        elif "research" in desc.lower() or agent == "research_agent":
            context += f"\nResearch (Step {i + 1}):\n{output.strip()}\n"
        else:
            context += f"\nOther (Step {i + 1}) by {agent}:\n{output.strip()}\n"

    enriched_task = f"""{context}

Your next task:
{step_title}
"""

    # Select agent based on the step
    step_lower = step_title.lower()
    if "research" in step_lower:
        content, _ = research_agent(prompt=enriched_task)
        print("Research Agent Output:", content)
        return step_title, "research_agent", content
    elif "draft" in step_lower or "write" in step_lower:
        content, _ = writer_agent(prompt=enriched_task)
        return step_title, "writer_agent", content
    elif "revise" in step_lower or "edit" in step_lower or "feedback" in step_lower:
        content, _ = editor_agent(prompt=enriched_task)
        return step_title, "editor_agent", content
    else:
        raise ValueError(f"Unknown step type: {step_title}")

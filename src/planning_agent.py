import json
import re
import ast
from typing import List
from datetime import datetime

from src.model_manager import get_planner_model
from src.agents import research_agent, writer_agent, editor_agent


def planner_agent(topic: str) -> List[str]:
    print("==================================")
    print(" Planner Agent (Mistral-7B)")
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
  
  **Topic:** {topic}
  
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

    print(f" Raw planner output: {raw[:300]}...")

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

from datetime import datetime
from typing import Tuple, List, Dict
import json
import re

from src.model_manager import (
    get_research_model,
    get_writer_model,
    get_editor_model
)
from src.research_tools import (
    arxiv_search_tool,
    duckduckgo_search_tool,
    wikipedia_search_tool,
    tool_mapping
)


def research_agent(
    prompt: str, 
    return_messages: bool = False
) -> Tuple[str, List[Dict]]:
    print("==================================")
    print("Research Agent")
    print("==================================")

    # Extract actual user query
    user_query = prompt.split("Your next task:")[-1].strip() if "Your next task:" in prompt else prompt
    user_query = user_query.split("Research agent:")[-1].strip() if "Research agent:" in user_query else user_query
    
    if "Use DuckDuckGo" in user_query or "search on arXiv" in user_query:
        # Extract topic from task description
        words = user_query.split()
        user_query = ' '.join(words[:5])  # First 5 words as topic
    
    full_prompt = f"""You are a research assistant. You must search for information about the user's topic.
USER'S TOPIC: {user_query}
Respond with EXACTLY this format:
ACTION: DUCKDUCKGO_SEARCH
INPUT: {user_query}
Start now:"""

    messages = [{"role": "user", "content": full_prompt}]
    model = get_research_model()
    
    tool_calls_made = []
    all_results = []
    
    # Just do ONE search and return results
    for iteration in range(2):
        response = model.chat_completion(messages, max_tokens=256, temperature=0.1)
        
        print(f"Response: {response[:150]}...")
        
        # Parse action
        action_match = re.search(r'ACTION[:\s]+(\w+)', response, re.IGNORECASE)
        input_match = re.search(r'INPUT[:\s]+(.+?)(?=\n|$)', response, re.IGNORECASE)
        
        if action_match and input_match:
            tool_name = action_match.group(1).upper()
            tool_input = input_match.group(1).strip()
            
            if "DUCKDUCKGO" in tool_name:
                try:
                    results = tool_mapping["duckduckgo_search_tool"](tool_input, max_results=5)
                    tool_calls_made.append(("duckduckgo_search_tool", tool_input))
                    all_results.extend(results)
                    
                    # Format results immediately
                    output = f"Search results for '{tool_input}':\n\n"
                    for i, r in enumerate(results[:5], 1):
                        output += f"{i}. {r.get('title', 'No title')}\n"
                        output += f"   {r.get('content', '')[:200]}...\n"
                        output += f"   URL: {r.get('url', 'N/A')}\n\n"
                    
                    # Add tools used section
                    output += "\n<h2>Tools used</h2><ul>"
                    output += f"<li>duckduckgo_search_tool({tool_input})</li>"
                    output += "</ul>"
                    
                    return output, messages
                    
                except Exception as e:
                    return f"Search error: {str(e)}", messages
        
        # If no valid action, try once more with explicit instruction
        if iteration == 0:
            messages.append({
                "role": "user",
                "content": f"ACTION: DUCKDUCKGO_SEARCH\nINPUT: {user_query}"
            })
    
    # Fallback
    if tool_calls_made:
        return f"Completed search for: {user_query}. Found {len(all_results)} results.", messages
    return f"Unable to search for: {user_query}", messages
    
# ===== Writer Agent =====
def writer_agent(
    prompt: str,
    min_words_total: int = 2400,
    min_words_per_section: int = 400,
    max_tokens: int = 4096,
) -> Tuple[str, List[Dict]]:
    print("==================================")
    print(" Writer Agent (Mixtral-8x7B)")
    print("==================================")

    system_message = """
You are an expert academic writer with PhD-level expertise. Produce a COMPLETE, POLISHED academic report in Markdown.
## MANDATORY STRUCTURE:
1. **Title**: Clear and descriptive
2. **Abstract**: 100-150 words summarizing purpose, methods, key findings
3. **Introduction**: Topic, research question, significance, outline
4. **Background/Literature Review**: Contextualize within existing scholarship
5. **Methodology**: Research methods and analytical approaches (if applicable)
6. **Key Findings/Results**: Primary outcomes and evidence
7. **Discussion**: Interpret findings, implications, limitations
8. **Conclusion**: Synthesize and suggest future directions
9. **References**: Complete list with URLs
## REQUIREMENTS:
- Use numeric inline citations [1], [2] for ALL borrowed information
- Every citation must have corresponding References entry
- Use HTML links with target="_blank": <a href="URL" target="_blank">text</a>
- Length: 1500-3000 words minimum
- Formal academic tone with discipline-appropriate terminology
- Original analysis beyond mere summarization
Output ONLY the complete Markdown report.
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    model = get_writer_model()
    
    content = model.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=0.3
    )

    print(" Draft completed")
    return content, messages


# ===== Editor Agent =====
def editor_agent(
    prompt: str,
    target_min_words: int = 2400,
) -> Tuple[str, List[Dict]]:
    print("==================================")
    print(" Editor Agent (Llama-3-8B)")
    print("==================================")

    system_message = """
You are a professional academic editor. Refine and elevate the scholarly text provided.
## Your Tasks:
1. Analyze structure, argument flow, and coherence
2. Ensure logical progression with clear transitions
3. Improve clarity and conciseness while maintaining academic tone
4. Strengthen thesis statements and main arguments
5. Verify proper integration of evidence
6. Standardize terminology and eliminate redundancies
7. Preserve ALL citations [1], [2] and References section integrity
8. Use HTML for all links: <a href="URL" target="_blank">text</a>
Return ONLY the revised Markdown text without meta-commentary.
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    model = get_editor_model()
    
    content = model.chat_completion(
        messages,
        max_tokens=4096,
        temperature=0.1
    )

    print(" Editing completed")
    return content, messages

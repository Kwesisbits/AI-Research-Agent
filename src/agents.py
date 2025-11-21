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


# ===== Research Agent with ReAct Pattern =====
def research_agent(
    prompt: str, 
    return_messages: bool = False
) -> Tuple[str, List[Dict]]:
    print("==================================")
    print(" Research Agent (Hermes-2-Pro)")
    print("==================================")

    full_prompt = f"""
You are an advanced research assistant with expertise in information retrieval. You have access to these tools:
**Available Tools:**
1. DUCKDUCKGO_SEARCH - General web search (news, blogs, websites, industry reports)
   Format: ACTION: DUCKDUCKGO_SEARCH
           INPUT: your search query
2. ARXIV_SEARCH - Academic papers (Computer Science, Math, Physics, Statistics only)
   Format: ACTION: ARXIV_SEARCH
           INPUT: your search terms
3. WIKIPEDIA_SEARCH - Encyclopedia for background information and definitions
   Format: ACTION: WIKIPEDIA_SEARCH
           INPUT: topic name
**Instructions:**
1. Analyze the user's research request carefully
2. Use appropriate tools to gather comprehensive information
3. After receiving tool results, analyze them and decide if you need more information
4. When you have sufficient information, provide your final answer starting with "FINAL_ANSWER:"
5. Include source URLs in your final answer
**Format Requirements:**
- To use a tool: Write exactly "ACTION: TOOL_NAME" on one line, then "INPUT: query" on the next
- After tool results: Either use another tool OR provide "FINAL_ANSWER: your response"
- Maximum 5 tool uses allowed
Today is {datetime.now().strftime("%Y-%m-%d")}.
**User Request:**
{prompt}
"""

    messages = [{"role": "user", "content": full_prompt}]
    
    model = get_research_model()
    
    conversation_history = []
    max_iterations = 5
    tool_calls_made = []
    
    for iteration in range(max_iterations):
        print(f"\n Iteration {iteration + 1}/{max_iterations}")
        
        response = model.chat_completion(
            messages,
            max_tokens=2048,
            temperature=0.1
        )
        
        print(f" Model Response Preview: {response[:200]}...")
        
        # Check for final answer
        if "FINAL_ANSWER:" in response:
            final_answer = response.split("FINAL_ANSWER:")[1].strip()
            
            # Append tool calls summary
            if tool_calls_made:
                tools_html = "<h2 style='font-size:1.5em; color:#2563eb;'> Tools used</h2><ul>"
                for tool_name, tool_input in tool_calls_made:
                    tools_html += f"<li>{tool_name}({tool_input})</li>"
                tools_html += "</ul>"
                final_answer += "\n\n" + tools_html
            
            print(" Research completed")
            return final_answer, messages
        
        # Parse ACTION and INPUT
        action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
        input_match = re.search(r'INPUT:\s*(.+?)(?=\n\n|\nACTION:|\n*$)', response, re.DOTALL | re.IGNORECASE)
        
        if action_match and input_match:
            tool_name = action_match.group(1).strip().lower()
            tool_input = input_match.group(1).strip()
            
            # Normalize tool name
            tool_map = {
                "duckduckgo_search": "duckduckgo_search_tool",
                "arxiv_search": "arxiv_search_tool",
                "wikipedia_search": "wikipedia_search_tool",
            }
            
            full_tool_name = tool_map.get(tool_name, tool_name + "_tool")
            
            if full_tool_name in tool_mapping:
                print(f" Calling {full_tool_name} with: {tool_input}")
                
                try:
                    tool_result = tool_mapping[full_tool_name](tool_input)
                    tool_calls_made.append((full_tool_name, tool_input))
                    
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": f"TOOL_RESULT from {full_tool_name}:\n{json.dumps(tool_result, indent=2)}\n\nAnalyze these results and either:\n1. Use another tool if you need more information\n2. Provide FINAL_ANSWER: with your comprehensive response"
                    })
                    
                    print(f" Tool executed, returned {len(tool_result)} results")
                except Exception as e:
                    print(f" Tool execution failed: {e}")
                    messages.append({
                        "role": "user",
                        "content": f"ERROR executing {full_tool_name}: {str(e)}\nTry a different tool or provide your answer based on previous results."
                    })
            else:
                print(f" Unknown tool: {tool_name}")
                messages.append({
                    "role": "user",
                    "content": f"ERROR: Unknown tool '{tool_name}'. Available tools:\n- DUCKDUCKGO_SEARCH\n- ARXIV_SEARCH\n- WIKIPEDIA_SEARCH\n\nPlease use correct tool name or provide FINAL_ANSWER:"
                })
        else:
            # Model didn't follow format
            print(" Model response doesn't match format")
            messages.append({
                "role": "user",
                "content": "Please follow the exact format:\n\nACTION: TOOL_NAME\nINPUT: your query\n\nOR provide:\n\nFINAL_ANSWER: your response\n\nDo not include explanations before the action."
            })
    
    # Iteration limit reached
    print(" Max iterations reached")
    return "Research incomplete: Maximum iteration limit reached. Please try a more specific query.", messages


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

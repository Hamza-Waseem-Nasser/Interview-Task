"""
LangGraph agent orchestration for the HR AI Assistant.

This module wires together the RAG and structured data tools into a
LangGraph StateGraph. The agent uses the LLM's native tool-calling
capabilities to decide which tool(s) to invoke for each question.

Architecture:
  ┌─────────┐    tool calls    ┌───────────┐
  │  Agent  │ ──────────────→ │   Tools   │
  │  (LLM)  │ ←────────────── │ (RAG/CSV) │
  └─────────┘   tool results   └───────────┘
       ↑                            
       │ user question + history    
       │                            
  ┌─────────┐                  
  │  Entry  │                  
  └─────────┘                  

Routing logic:
  - The LLM decides which tool(s) to call based on the question
  - System prompt guides tool selection with clear instructions
  - If no tool is needed, the agent responds directly
  - "I don't know" for questions outside the data scope
"""

import json
import logging
import re
from typing import Literal

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from src.config import get_llm
from src.rag import search_policies, get_last_references as get_last_rag_refs
from src.structured_data import query_employee_data, get_last_references as get_last_employee_refs
from src.memory import get_session_history, add_to_session

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an internal HR AI Assistant for AlNoor Technologies. You help employees with HR-related questions using two data sources and two tools.

You MUST respond in the SAME LANGUAGE the employee uses. If they write in Arabic, respond in Arabic. If they write in English, respond in English. If they mix languages, match their primary language.

## CRITICAL: Tool Selection Rules

You have exactly two tools. You MUST follow these rules:

### Tool 1: `query_employee_data`
Call this tool when the question involves ANY personal/individual information.
TRIGGER WORDS (English): "my", "I", "me", "do I have", "am I", "my balance", "my budget", "my rating", "my department", "how many days", "how much budget", "remaining", "left"
TRIGGER WORDS (Arabic): "كم", "إجازة", "إجازاتي", "رصيدي", "ميزانيتي", "تقييمي", "قسمي", "مديري", "أنا", "لي", "عندي", "متبقي"
Examples that REQUIRE this tool:
  - "How many leave days do I have left?" → query_employee_data
  - "كم يوم إجازة متبقي لدي؟" → query_employee_data
  - "What is my performance rating?" → query_employee_data
  - "Who is my manager?" → query_employee_data

### Tool 2: `search_policies`
Call this tool when the question is about general company rules, policies, or procedures.
TRIGGER WORDS (English): "policy", "rule", "allowed", "limit", "procedure", "guidelines", "carry-over", "what happens if"
TRIGGER WORDS (Arabic): "سياسة", "قانون", "مسموح", "حد", "إجراء", "إرشادات", "ترحيل", "ماذا لو"
Examples that REQUIRE this tool:
  - "What is the carry-over limit for annual leave?" → search_policies
  - "ما هي سياسة العمل عن بعد؟" → search_policies

### BOTH TOOLS (very important):
Call BOTH tools when the question combines personal data with policy rules.
Examples that REQUIRE BOTH tools:
  - "Am I eligible for remote work?" → FIRST query_employee_data, THEN search_policies
  - "هل يمكنني العمل عن بعد؟" → FIRST query_employee_data, THEN search_policies
  - "How much training budget do I have left and what can I spend it on?" → FIRST query_employee_data, THEN search_policies
  - "Can I carry over my unused leave?" → FIRST query_employee_data, THEN search_policies

### NO TOOLS (respond directly):
If the question is:
- A greeting ("hi", "hello", "مرحبا", "how are you") → Respond: "Hello! I'm your HR AI Assistant at AlNoor Technologies. I can help you with leave balances, company policies, remote work eligibility, and more. How can I help you today?"
- Gibberish or unclear → Respond: "I didn't quite understand that. Could you please rephrase your HR-related question? I can help with leave balance, policies, remote work, training budget, and more."
- Outside HR scope (stock prices, weather, etc.) → Respond: "I don't have enough information to answer that question. This falls outside the scope of the HR policies and employee data I have access to." (If the user asks in Arabic, translate this refusal to Arabic, but DO NOT include any English meta-commentary like 'Since your question is...'. Just refuse directly).

## Response Guidelines

- Be concise, accurate, and helpful.
- ALWAYS respond in the same language the employee used.
- Always cite which source your answer comes from (policy document or employee database).
- If the employee is on probation or PIP, factor that into eligibility answers.
- Never make up information. Only use what the tools return.
- When presenting numerical data (leave days, budgets), be precise.
- When you cite a policy, mention the specific policy name and section.
- DO NOT output raw JSON or tool call descriptions. Only output natural language answers.

## Current Context
The employee asking the question will be identified by their employee_id. Always use this ID when querying employee data.
"""


# ── Helpers: clean LLM output and parse text-based tool calls ────────────────

def _clean_response(text: str | list) -> str:
    """
    Clean LLM response text:
    1. Handle LangChain list contents (Gemini often returns list of blocks)
    2. Remove <think>...</think> blocks (qwen3 internal reasoning)
    3. Remove raw JSON tool-call blobs the LLM sometimes outputs as text
    4. Strip whitespace
    """
    # LangChain Google GenAI sometimes returns content as a list of dictionaries
    if isinstance(text, list):
        text_parts = []
        for block in text:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        text = " ".join(text_parts)
    elif not isinstance(text, str):
        text = str(text)

    # Remove <think> blocks (qwen3 internal reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove JSON-like tool call blobs (e.g. {"name": "respond_directly", "parameters": {}})
    # These appear when the LLM outputs tool calls as text instead of structured format
    text = re.sub(
        r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*(?:"parameters"|"arguments")\s*:\s*\{[^{}]*\}[^{}]*\}',
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove standalone JSON blobs that are clearly not natural language
    text = re.sub(r'^\s*\{[^{}]*\}\s*$', "", text, flags=re.DOTALL)

    return text.strip()


def _extract_text_tool_calls(text: str) -> list[dict] | None:
    """
    Try to parse tool calls from LLM output when the model emits them
    as plain text instead of using structured tool-calling format.

    This handles cases where the LLM outputs something like:
      {"name": "query_employee_data", "parameters": {"employee_id": "EMP001"}}
    """
    # Common patterns for text-based tool calls
    patterns = [
        # JSON object with "name" and "parameters"
        r'\{[^{}]*"name"\s*:\s*"(search_policies|query_employee_data)"[^{}]*"parameters"\s*:\s*\{[^{}]*\}[^{}]*\}',
        # JSON object with "name" and "arguments"
        r'\{[^{}]*"name"\s*:\s*"(search_policies|query_employee_data)"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Try to extract full JSON objects
            tool_calls = []
            for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"(?:search_policies|query_employee_data)"[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL):
                try:
                    obj = json.loads(match.group())
                    name = obj.get("name", "")
                    args = obj.get("parameters", obj.get("arguments", {}))
                    if name in ("search_policies", "query_employee_data"):
                        tool_calls.append({"name": name, "args": args})
                except json.JSONDecodeError:
                    continue
            if tool_calls:
                return tool_calls

    return None


# ── Build the agent graph ────────────────────────────────────────────────────
def _build_graph():
    """Build and compile the LangGraph agent."""
    tools = [search_policies, query_employee_data]
    tool_map = {t.name: t for t in tools}
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        """The agent node: invokes the LLM with tools bound."""
        response = llm_with_tools.invoke(state["messages"])

        # --- Fallback: detect text-based tool calls BEFORE cleaning ---
        # Some models (especially smaller ones) output tool calls as plain text
        # instead of using the structured tool_calls property
        raw_content = ""
        if hasattr(response, "content") and isinstance(response.content, str):
            raw_content = response.content

        has_structured_calls = hasattr(response, "tool_calls") and response.tool_calls

        if raw_content and not has_structured_calls:
            text_tool_calls = _extract_text_tool_calls(raw_content)
            if text_tool_calls:
                logger.info(f"Detected text-based tool call(s), executing manually: "
                            f"{[tc['name'] for tc in text_tool_calls]}")
                # Execute each tool call manually
                tool_results = []
                for tc in text_tool_calls:
                    tool_func = tool_map.get(tc["name"])
                    if tool_func:
                        try:
                            result = tool_func.invoke(tc["args"])
                            tool_results.append(ToolMessage(
                                content=str(result),
                                tool_call_id=f"manual_{tc['name']}",
                                name=tc["name"],
                            ))
                        except Exception as e:
                            logger.error(f"Manual tool call failed: {e}")
                            tool_results.append(ToolMessage(
                                content=f"Error calling {tc['name']}: {e}",
                                tool_call_id=f"manual_{tc['name']}",
                                name=tc["name"],
                            ))

                if tool_results:
                    # Re-invoke the LLM with the tool results
                    tool_results_text = "\n".join(
                        f"[Tool: {m.name}]\n{m.content}" for m in tool_results
                    )
                    synthesis_msg = HumanMessage(
                        content=f"Here are the results from the tools I called:\n\n"
                                f"{tool_results_text}\n\n"
                                f"Please provide a clear, helpful answer to the employee's question based on these results. "
                                f"Do NOT output any JSON or tool calls. Only output a natural language answer. "
                                f"Respond strictly in the SAME language the employee used. DO NOT mix English reasoning with Arabic output."
                    )
                    synth_messages = [msg for msg in state["messages"]] + [synthesis_msg]
                    synthesis_response = llm.invoke(synth_messages)
                    if hasattr(synthesis_response, "content"):
                        synthesis_response.content = _clean_response(synthesis_response.content)
                    synthesis_response._manual_tools_called = [tc["name"] for tc in text_tool_calls]
                    return {"messages": [synthesis_response]}

        # Clean the response text (remove <think> blocks, JSON blobs, etc.)
        if hasattr(response, "content") and isinstance(response.content, str):
            response.content = _clean_response(response.content)

            # If the response is empty after cleaning (was purely JSON),
            # provide a sensible default greeting
            if not response.content and not has_structured_calls:
                logger.info("Response was empty after cleaning (likely JSON-only output). Using fallback.")
                response.content = (
                    "Hello! I'm your HR AI Assistant at AlNoor Technologies. "
                    "I can help you with:\n\n"
                    "• Leave balance and policies\n"
                    "• Remote work eligibility\n"
                    "• Performance reviews\n"
                    "• Training budget\n"
                    "• Company policies and procedures\n\n"
                    "How can I help you today?"
                )

        return {"messages": [response]}


    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        """Route to tools if the LLM made tool calls, otherwise end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    # Build the graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Module-level compiled graph (singleton) ──────────────────────────────────
_compiled_graph = None


def _get_graph():
    """Lazy-initialize the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Building LangGraph agent...")
        _compiled_graph = _build_graph()
        logger.info("Agent graph compiled successfully.")
    return _compiled_graph


def determine_source(messages: list[BaseMessage]) -> str:
    """
    Determine which source(s) were used based on tool calls in the message history.

    Returns one of: "rag", "structured_data", "both", "unknown"
    """
    tools_used = set()

    for msg in messages:
        # Check structured tool_calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "")
                if tool_name == "search_policies":
                    tools_used.add("rag")
                elif tool_name == "query_employee_data":
                    tools_used.add("structured_data")

        # Check for manually executed tool calls (fallback path)
        if hasattr(msg, "_manual_tools_called"):
            for tool_name in msg._manual_tools_called:
                if tool_name == "search_policies":
                    tools_used.add("rag")
                elif tool_name == "query_employee_data":
                    tools_used.add("structured_data")

    if tools_used == {"rag", "structured_data"}:
        return "both"
    elif "rag" in tools_used:
        return "rag"
    elif "structured_data" in tools_used:
        return "structured_data"
    return "unknown"


async def ask_agent(
    employee_id: str,
    question: str,
    session_id: str | None = None,
) -> dict:
    """
    Send a question to the HR AI Agent.

    Args:
        employee_id: The employee's ID (e.g., "EMP001").
        question: The employee's question.
        session_id: Optional session ID for conversation memory.

    Returns:
        dict with keys: "answer", "source", "session_id"
    """
    graph = _get_graph()

    # Build the message list
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add session history if available
    if session_id:
        history = get_session_history(session_id)
        messages.extend(history)

    # Add the current question with employee context
    user_msg = HumanMessage(
        content=f"[Employee ID: {employee_id}]\n\n{question}"
    )
    messages.append(user_msg)

    # Invoke the graph
    logger.info(f"Processing question from {employee_id}: {question[:100]}...")
    result = await graph.ainvoke({"messages": messages})

    # --- RAW LOGGING FOR INVESTIGATION ---
    print("\n" + "="*60)
    print("🔍 RAW INVESTIGATION LOGS")
    print("="*60)
    for i, msg in enumerate(result["messages"]):
        # Skip the massive system prompt for cleaner logs
        if type(msg).__name__ == "SystemMessage":
            print(f"\n[{i}] SystemMessage (Hidden to save space)")
            continue
            
        print(f"\n[{i}] TYPE: {type(msg).__name__}")
        
        # Log tool calls if AI message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"🔧 LLM GENERATED TOOL CALLS:\n{json.dumps(msg.tool_calls, indent=2)}")
            
        # Log content
        content = getattr(msg, "content", str(msg))
        if content:
            # truncate long content slightly for readability
            if len(content) > 1000:
                print(f"📄 CONTENT:\n{content[:1000]}... [TRUNCATED]")
            else:
                print(f"📄 CONTENT:\n{content}")
            
        # Log manual fallback tools
        if hasattr(msg, "_manual_tools_called") and msg._manual_tools_called:
            print(f"⚠️ MANUAL FALLBACK TOOLS TRIGGERED: {msg._manual_tools_called}")
            
        # Log Tool Message specifics
        if type(msg).__name__ == "ToolMessage":
            print(f"🛠️  TOOL EXECUTED: {getattr(msg, 'name', 'unknown')}")
    print("="*60 + "\n")

    # Extract the final answer
    all_messages = result["messages"]
    final_message = all_messages[-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    # Clean the answer one more time
    answer = _clean_response(answer)

    # Determine source
    source = determine_source(all_messages)

    # Collect references
    references = []
    if source in ("rag", "both"):
        references.extend(get_last_rag_refs())
    if source in ("structured_data", "both"):
        references.extend(get_last_employee_refs())

    # Save to session memory
    if session_id:
        # Save only the user message and the final AI response
        add_to_session(session_id, [user_msg, AIMessage(content=answer)])

    logger.info(f"Response source: {source}, references: {len(references)}")
    return {
        "answer": answer,
        "source": source,
        "references": references,
        "session_id": session_id,
    }

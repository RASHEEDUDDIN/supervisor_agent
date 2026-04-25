"""
Multi-agent supervisor pipeline — LangGraph + Groq (FREE)
==========================================================
Setup:
  pip install langgraph langchain-groq langchain-core

Run:
  export GROQ_API_KEY=gsk_...
  python supervisor_agent_groq.py
"""

import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler


langfuse_handler = CallbackHandler()
# ─────────────────────────────────────────────
# 1.  LLM  (free, 315 tokens/sec on Llama 3.3 70B)
# ─────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    max_tokens=1024,
)


# ─────────────────────────────────────────────
# 2.  Shared state
# ─────────────────────────────────────────────
class State(TypedDict):
    task: str
    analysis: str
    tests: str
    next: str
    steps: int


# ─────────────────────────────────────────────
# 3.  Nodes
# ─────────────────────────────────────────────

def supervisor_node(state: State) -> dict:
    done = []
    if state.get("analysis"):
        done.append("code_analyzer has run")
    if state.get("tests"):
        done.append("test_generator has run")

    progress = ", ".join(done) if done else "nothing done yet"

    prompt = f"""You coordinate two workers on a software testing task.

Workers available:
  code_analyzer   — identifies edge cases and testing requirements in code
  test_generator  — writes pytest tests (requires code_analyzer output first)

Task:
{state["task"]}

Progress: {progress}

Reply with ONLY one of these exact words (no punctuation, no explanation):
  code_analyzer
  test_generator
  FINISH"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    if "code_analyzer" in raw:
        decision = "code_analyzer"
    elif "test_generator" in raw:
        decision = "test_generator"
    else:
        decision = "FINISH"

    print(f"  [Supervisor]     → {decision}")
    return {"next": decision, "steps": state.get("steps", 0) + 1}


def code_analyzer_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior QA engineer. "
            "Analyze Python code and list every case that needs pytest coverage. "
            "Be concise — bullet points only."
        )),
        HumanMessage(content=(
            f"Analyze this function and list every test case needed:\n\n{state['task']}"
        )),
    ])
    print(f"  [CodeAnalyzer]   done ({len(response.content)} chars)")
    return {"analysis": response.content}


def test_generator_node(state: State) -> dict:
    context = f"Function to test:\n{state['task']}"
    if state.get("analysis"):
        context += f"\n\nRequired test cases:\n{state['analysis']}"

    response = llm.invoke([
        SystemMessage(content=(
            "You are a pytest expert. "
            "Output ONLY valid Python pytest code — no markdown fences, no prose. "
            "Use descriptive test function names. Include edge cases."
        )),
        HumanMessage(content=context),
    ])
    print(f"  [TestGenerator]  done ({len(response.content)} chars)")
    return {"tests": response.content}


# ─────────────────────────────────────────────
# 4.  Routing
# ─────────────────────────────────────────────

def route(state: State) -> Literal["code_analyzer", "test_generator", "end"]:
    if state.get("steps", 0) >= 6:
        return "end"
    decision = state.get("next", "FINISH")
    if decision == "code_analyzer":
        return "code_analyzer"
    if decision == "test_generator":
        return "test_generator"
    return "end"


# ─────────────────────────────────────────────
# 5.  Build graph
# ─────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("supervisor",     supervisor_node)
builder.add_node("code_analyzer",  code_analyzer_node)
builder.add_node("test_generator", test_generator_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route,
    {
        "code_analyzer":  "code_analyzer",
        "test_generator": "test_generator",
        "end":            END,
    },
)

builder.add_edge("code_analyzer",  "supervisor")
builder.add_edge("test_generator", "supervisor")

app = builder.compile()


# ─────────────────────────────────────────────
# 6.  Run
# ─────────────────────────────────────────────

TASK = '''
def calculate_discount(price: float, discount_pct: float) -> float:
    """Apply a percentage discount to a price and return the final amount."""
    if not isinstance(price, (int, float)):
        raise TypeError("price must be numeric")
    if discount_pct < 0 or discount_pct > 100:
        raise ValueError(f"discount_pct must be 0-100, got {discount_pct}")
    return round(price * (1 - discount_pct / 100), 2)
'''

if __name__ == "__main__":
    print("\nStarting pipeline (powered by Groq + Llama 3.3 70B)...\n")

    result = app.invoke(
        {"task": TASK, "analysis": "", "tests": "", "next": "", "steps": 0},
        config={"callbacks": [langfuse_handler]},
    )

    print("\n" + "=" * 60)
    print("CODE ANALYSIS")
    print("=" * 60)
    print(result["analysis"])

    print("\n" + "=" * 60)
    print("GENERATED PYTEST TESTS")
    print("=" * 60)
    print(result["tests"])

    print(f"\nCompleted in {result['steps']} supervisor decisions.")


    with open("test_generated.py", "w") as f:
        # strip the markdown fences Groq sometimes adds
        code = result["tests"].replace("```python", "").replace("```", "").strip()
        f.write("from discount import calculate_discount\n\n")
        f.write(code)
    print("\nTests saved to test_generated.py")
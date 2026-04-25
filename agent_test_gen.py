"""
Agentic Test Generation Pipeline — reads ANY Python file
=========================================================
Setup:
  pip install langgraph langchain-groq langchain-core python-dotenv

Usage:
  python agent_test_gen.py discount.py
  python agent_test_gen.py my_module.py

Output:
  test_<filename>.py  — runnable pytest file, auto-saved
"""

import os
import ast
import sys
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 1.  LLM
# ─────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    max_tokens=2048,
)


# ─────────────────────────────────────────────
# 2.  AST parser — extracts every function from a .py file
# ─────────────────────────────────────────────

def extract_functions(filepath: str) -> list[dict]:
    """
    Parse a Python file and return a list of dicts, one per function:
      { name, source, args, returns, docstring }
    """
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    lines = source.splitlines()
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # extract raw source lines for this function
            start = node.lineno - 1
            end = node.end_lineno
            func_source = "\n".join(lines[start:end])

            # extract argument names
            args = [arg.arg for arg in node.args.args]

            # extract return annotation if present
            returns = ""
            if node.returns:
                returns = ast.unparse(node.returns)

            # extract docstring if present
            docstring = ast.get_docstring(node) or ""

            functions.append({
                "name":      node.name,
                "source":    func_source,
                "args":      args,
                "returns":   returns,
                "docstring": docstring,
            })

    return functions


# ─────────────────────────────────────────────
# 3.  Shared state
# ─────────────────────────────────────────────

class State(TypedDict):
    filepath: str          # input file path
    module_name: str       # e.g. "discount" from "discount.py"
    functions: list        # parsed function dicts from ast
    current_fn: dict       # the function being worked on right now
    analysis: str          # code_analyzer output
    tests: str             # test_generator output
    all_tests: list        # accumulated tests for all functions
    fn_index: int          # which function we're on
    next: str              # supervisor routing decision
    steps: int


# ─────────────────────────────────────────────
# 4.  Nodes
# ─────────────────────────────────────────────

def supervisor_node(state: State) -> dict:
    fn_index = state.get("fn_index", 0)
    functions = state.get("functions", [])
    analysis  = state.get("analysis", "")
    tests     = state.get("tests", "")

    # If tests were just generated, save them and advance to next function
    if tests:
        all_tests = state.get("all_tests", [])
        all_tests.append(tests)
        fn_index += 1

        # Reset per-function state
        if fn_index >= len(functions):
            print(f"  [Supervisor]     → FINISH (all {len(functions)} functions done)")
            return {
                "next":      "FINISH",
                "fn_index":  fn_index,
                "all_tests": all_tests,
                "analysis":  "",
                "tests":     "",
            }

        # Move to next function
        current_fn = functions[fn_index]
        print(f"  [Supervisor]     → code_analyzer (fn: {current_fn['name']})")
        return {
            "next":       "code_analyzer",
            "fn_index":   fn_index,
            "current_fn": current_fn,
            "all_tests":  all_tests,
            "analysis":   "",
            "tests":      "",
            "steps":      state.get("steps", 0) + 1,
        }

    # Analysis done, now generate tests
    if analysis and not tests:
        print(f"  [Supervisor]     → test_generator")
        return {"next": "test_generator", "steps": state.get("steps", 0) + 1}

    # Nothing done yet — start with first function
    if functions:
        current_fn = functions[fn_index]
        print(f"  [Supervisor]     → code_analyzer (fn: {current_fn['name']})")
        return {
            "next":       "code_analyzer",
            "current_fn": current_fn,
            "steps":      state.get("steps", 0) + 1,
        }

    return {"next": "FINISH"}


def code_analyzer_node(state: State) -> dict:
    fn = state["current_fn"]
    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior QA engineer. "
            "Analyze a Python function and list every case that needs pytest coverage. "
            "Be concise — bullet points only. No prose."
        )),
        HumanMessage(content=(
            f"Function name: {fn['name']}\n"
            f"Arguments: {fn['args']}\n"
            f"Returns: {fn['returns']}\n"
            f"Docstring: {fn['docstring']}\n\n"
            f"Source:\n{fn['source']}\n\n"
            f"List every test case needed:"
        )),
    ])
    print(f"  [CodeAnalyzer]   done — {fn['name']} ({len(response.content)} chars)")
    return {"analysis": response.content}


def test_generator_node(state: State) -> dict:
    fn = state["current_fn"]
    module = state["module_name"]

    response = llm.invoke([
        SystemMessage(content=(
            "You are a pytest expert. "
            f"Write tests that import from '{module}'. "
            "Output ONLY valid Python pytest code — no markdown fences, no prose, no import statements. "
            "Use descriptive test function names."
        )),
        HumanMessage(content=(
            f"Function source:\n{fn['source']}\n\n"
            f"Required test cases:\n{state['analysis']}\n\n"
            f"Write pytest tests for {fn['name']}:"
        )),
    ])
    # Strip markdown fences if model adds them
    raw = response.content.replace("```python", "").replace("```", "").strip()
    print(f"  [TestGenerator]  done — {fn['name']} ({len(raw)} chars)")
    return {"tests": raw}


# ─────────────────────────────────────────────
# 5.  Routing
# ─────────────────────────────────────────────

def route(state: State) -> Literal["code_analyzer", "test_generator", "end"]:
    if state.get("steps", 0) >= 50:
        return "end"
    decision = state.get("next", "FINISH")
    if decision == "code_analyzer":
        return "code_analyzer"
    if decision == "test_generator":
        return "test_generator"
    return "end"


# ─────────────────────────────────────────────
# 6.  Build graph
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
# 7.  Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Accept file path from command line, default to discount.py
    filepath = sys.argv[1] if len(sys.argv) > 1 else "discount.py"

    if not os.path.exists(filepath):
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)

    module_name = os.path.splitext(os.path.basename(filepath))[0]
    functions   = extract_functions(filepath)

    if not functions:
        print(f"No functions found in {filepath}")
        sys.exit(1)

    print(f"\nFound {len(functions)} function(s) in {filepath}:")
    for fn in functions:
        print(f"  - {fn['name']}({', '.join(fn['args'])})")

    print(f"\nStarting pipeline...\n")

    result = app.invoke({
        "filepath":    filepath,
        "module_name": module_name,
        "functions":   functions,
        "current_fn":  functions[0],
        "analysis":    "",
        "tests":       "",
        "all_tests":   [],
        "fn_index":    0,
        "next":        "",
        "steps":       0,
    })

    # Assemble final test file
    output_file = f"test_{module_name}.py"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"import pytest\n")
        f.write(f"from {module_name} import *\n\n")
        for test_block in result["all_tests"]:
            f.write(test_block.strip())
            f.write("\n\n")

    print(f"\nAll tests saved to {output_file}")
    print(f"Run with: pytest {output_file} -v")

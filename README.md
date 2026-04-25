# Agentic Test Generation Pipeline

A multi-agent AI system that autonomously analyzes Python code, identifies test cases, and generates runnable pytest suites — orchestrated with **LangGraph**, powered by **Groq + Llama 3.3 70B** (free), and traced with **Langfuse**.

---

## Architecture

```
START
  │
  ▼
SupervisorAgent  ──► CodeAnalyzer Agent  ──┐
  ▲                                         │
  │◄────────────────────────────────────────┘
  │
  ├──► TestGenerator Agent  ──┐
  ▲                            │
  │◄───────────────────────────┘
  │
  ▼
FINISH  →  test_discount.py written to disk
```

The **supervisor** is the only node that reasons about routing — it reads shared state and decides which worker runs next. Workers are stateless: they receive the state, do one job, and hand back to the supervisor. The loop ends when the supervisor outputs `FINISH`.

---

## What it does

1. Supervisor reads the target function and routes to `code_analyzer`
2. `code_analyzer` calls Llama 3.3 70B to identify edge cases, boundary values, and error conditions
3. Supervisor sees the analysis is complete and routes to `test_generator`
4. `test_generator` calls Llama 3.3 70B to write a full pytest suite based on the analysis
5. Supervisor detects both workers have run and outputs `FINISH`
6. Generated tests are saved to `test_discount.py` automatically
7. Every LLM call is traced in Langfuse (latency, token count, cost)

**Total runtime: ~10 seconds. Total API cost: $0.00 (Groq free tier).**

---

## Sample output

```
Starting pipeline (Groq + Llama 3.3 70B + Langfuse tracing)...

  [Supervisor]     → code_analyzer
  [CodeAnalyzer]   done (575 chars)
  [Supervisor]     → test_generator
  [TestGenerator]  done (1801 chars)
  [Supervisor]     → FINISH

Completed in 3 supervisor decisions.
Tests saved to test_discount.py
```

Generated tests pass at **11/12** (one case exposes a Python banker's rounding edge — documented below).

---

## Tech stack

| Layer | Tool |
|---|---|
| Agent orchestration | LangGraph 0.2+ |
| LLM | Groq — Llama 3.3 70B Versatile |
| Observability | Langfuse (full trace per run) |
| Test framework | pytest |
| Language | Python 3.12 |

---

## Project structure

```
langgraph-demo/
  supervisor_agent_groq.py  # LangGraph graph — supervisor + 2 worker agents
  discount.py               # target function under test
  test_discount.py          # auto-generated pytest suite (output)
  .gitignore
  README.md
```

---

## Setup

**1. Clone and create virtual environment**
```bash
git clone https://github.com/RASHEEDUDDIN/supervisor_agent.git
cd supervisor_agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
```

**2. Install dependencies**
```bash
pip install langgraph langchain-groq langchain-core langfuse
```

**3. Get free API keys**

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free, no credit card |
| `LANGFUSE_PUBLIC_KEY` | [cloud.langfuse.com](https://cloud.langfuse.com) — free tier |
| `LANGFUSE_SECRET_KEY` | Same as above |

**4. Set environment variables**
```bash
# Mac/Linux
export GROQ_API_KEY=gsk_...
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

# Windows PowerShell
$env:GROQ_API_KEY="gsk_..."
$env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$env:LANGFUSE_SECRET_KEY="sk-lf-..."
```

**5. Run**
```bash
python supervisor_agent_groq.py
```

**6. Run generated tests**
```bash
pytest test_discount.py -v
```

---

## Observability

Every pipeline run creates a trace in Langfuse showing the full agent execution:

```
▼ pipeline run  (~3s total)
  ├── supervisor     312 tokens  →  "code_analyzer"
  ├── code_analyzer  587 tokens  →  bullet list of test cases
  ├── supervisor     401 tokens  →  "test_generator"
  ├── test_generator 1801 tokens →  pytest code
  └── supervisor     298 tokens  →  "FINISH"
```

Open `cloud.langfuse.com` → your project → Traces after each run.

---

## Key design decisions

**Why LangGraph over AutoGen?**
LangGraph's conditional edge model gives deterministic routing — the supervisor's routing decision is a Python function, not an LLM free-form conversation. This makes the system predictable and debuggable.

**Why Groq?**
315 tokens/second on Llama 3.3 70B, free tier, OpenAI-compatible endpoint. For a demo pipeline consuming ~3,400 tokens per run, the free tier handles hundreds of runs before hitting limits.

**The banker's rounding finding**
The AI-generated test `assert calculate_discount(100.0, 10.005) == 89.99` fails because Python's `round()` uses banker's rounding — `89.995` rounds to `90.0` (nearest even), not `89.99`. This is correct behavior in the function; the generated test had a wrong assumption. This demonstrates the need for human-in-the-loop review of AI-generated tests.

---

## Relevance to agentic AI roles

This project directly demonstrates:
- **Multi-agent orchestration** — supervisor pattern with conditional routing
- **LangGraph state management** — shared TypedDict state flowing between nodes
- **Agentic evaluation infrastructure** — Langfuse tracing per LLM call
- **Automated test generation** — LLM-driven pytest suite creation
- **CI-ready output** — generated tests runnable immediately with `pytest`

---

## Author

Rasheeduddin Mohammed (Ghouse) — [github.com/RASHEEDUDDIN](https://github.com/RASHEEDUDDIN)

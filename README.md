# Local Autonomous Research Agent

## Project Overview

This project builds a **fully local autonomous research agent** in Python with zero external APIs. The agent is designed to:

- Accept a research topic or question.
- Generate a research plan.
- Reason iteratively using a **ReAct-style loop**.
- Call only **local or simulated tools**.
- Maintain short-term memory and **long-term vector memory using FAISS**.
- Include a **reflection/critique phase**.
- Produce a structured JSON final report.
- Enforce strict iteration limits and guardrails.

**Constraints:**

- No cloud services or paid APIs.  
- Python-only, running fully locally via Ollama (LLM) + FAISS.  
- No LangChain, AutoGen, or high-level agent frameworks initially.  
- Manual implementation of ReAct loop, tool abstraction, parser, memory injection, logging, and deterministic control.

---

## Project Architecture

**High-Level Flow:**
```mermaid
flowchart TD

%% Core Components
LLM[Local LLM - Generates THOUGHT + ACTION/FINAL]
Parser[Parser - Validates schema & rejects malformed outputs]
Runtime[Runtime / Loop - Iteration tracking, max cap, guards, logging]
Tools[Tool Layer - Executes ACTIONs on local tools / simulations]
AgentState[AgentState - Maintains ST memory & injects LT memory]
FAISS[Long-Term Memory - FAISS vector store]
Reflection[Reflection / Critique - Evaluates previous steps]
FinalReport[Final Report Generator - Structured JSON output]

%% Data Flow
LLM --> Parser --> Runtime --> AgentState --> Tools --> AgentState
AgentState --> FAISS
AgentState --> Reflection --> AgentState
AgentState --> FinalReport
FinalReport --> Output[Structured JSON Report]

%% Notes (optional)
LLM --- LLMNote[Insight: LLM is policy engine; cannot self-enforce multi-step reasoning]
Runtime --- RuntimeNote[Lesson: Runtime ensures determinism and safe loop control]
Tools --- ToolsNote[Note: Tools provide actionable environment; essential for multi-step reasoning]
AgentState --- StateNote[Lesson: Centralized state supports deterministic transitions and memory injection]
FAISS --- FAISSNote[Insight: Long-term vector memory persists knowledge across iterations]
Reflection --- ReflectionNote[Lesson: Reflection step improves reasoning & error mitigation]
FinalReport --- ReportNote[Insight: Structured output enforces schema & determinism]
```


**Key Principles:**

- **Separation of concerns:**  
  - LLM = policy engine  
  - Parser = validator  
  - Runtime/Loop = controller/enforcer  
  - AgentState = deterministic memory/state  

- **Minimal prompt + runtime enforcement** ensures stability.  
- **Tool/environment dependencies** are required to motivate multi-step reasoning.  
- **Parse errors and rejected outputs** are informative signals for agent design.

---

## Daily Progress Tracker

### Day 1 – Setup, Loop & Schema Validation

**Objectives Completed:**

- Installed and tested Ollama locally with 3B LLM.
- Implemented deterministic agent loop with iteration tracking, max iteration cap, logging, and AgentState.
- Defined ReAct-style structured output schema: THOUGHT + ACTION or THOUGHT + FINAL; parser enforces mutual exclusivity.
- Added runtime guard to reject premature FINAL.
- Tested and simplified prompts for stable compliance.

**Key Lessons:**

- Small models often shortcut to FINAL; cannot reliably enforce multi-step reasoning in prompt.  
- Runtime enforcement is critical; prompt alone is fragile.  
- Over-constraining prompts destabilizes outputs; minimal prompt + runtime guard is more robust.  
- Actions will only become meaningful when tools/observations exist.  
- Deterministic loop, parser, and AgentState provide safe, reproducible control.

**Day 1 Cheat Sheet:**
```mermaid
flowchart TD

%% Core Components
A[LLM 3B: Predicts THOUGHT & ACTION/FINAL]
B[Parser: Validates schema & rejects malformed outputs]
C[Runtime/Loop: Iteration tracking, max cap, premature FINAL guard]
D[AgentState: Tracks iterations & short/long-term memory]
E[Output: FINAL or ACTION+THOUGHT]

%% Insights / Lessons as note nodes
AN[Insight: Small model shortcuts to FINAL if unconstrained]
BN[Lesson: Parser ensures deterministic agent steps]
CN[Insight: Runtime > prompt for enforcing rules; prevents premature FINAL]
DN[Note: Central state enables deterministic transitions and future memory/tool integration]
EN[Lesson: Actions meaningful only with tools/observations]

%% Connections
A --> B --> C --> D --> E

%% Connect lessons to components with dashed lines
A --- AN
B --- BN
C --- CN
D --- DN
E --- EN
```
---
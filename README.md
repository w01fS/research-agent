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
LLM[Local LLM - Generates JSON with THOUGHT + Action/Final]
Parser[Strict JSON Parser - Validates schema & returns dict]
FSM[FSM Runtime / Loop - Transitions state & detects loops]
Tools[Tool Layer - Executes ACTIONs on local tools / simulations]
AgentState[AgentState - Deterministic memory & state container]
FAISS[Long-Term Memory - FAISS vector store]
Reflection[Reflection / Critique - Evaluates previous steps]
FinalReport[Final Report Generator - Structured JSON output]

%% Data Flow
LLM --> Parser --> FSM --> AgentState --> Tools --> AgentState
AgentState --> FAISS
AgentState --> Reflection --> AgentState
AgentState --> FinalReport
FinalReport --> Output[Structured JSON Report]

%% Notes (optional)
LLM --- LLMNote[Insight: Large context windows benefit from 'History at End' prompt patterns]
Parser --- ParserNote[Lesson: Strict JSON parsing eliminates regex-based 'fuzzy' failures]
FSM --- FSMNote[Runtime: Deterministic transitions provide safe loop control and cycle detection]
AgentState --- StateNote[Architecture: State decoupling makes agent logic unit-testable]
FAISS --- FAISSNote[Insight: Long-term vector memory persists knowledge across iterations]
Reflection --- ReflectionNote[Lesson: Reflection step improves reasoning & error mitigation]
FinalReport --- ReportNote[Insight: Structured output enforces schema & determinism]
```


**Key Principles:**

- **Separation of concerns:**  
  - LLM = policy engine  
  - Parser = validator  
  - FSM Transition = controller/state logic  
  - AgentState = deterministic data container  

- **Guardrail Principles:**
  - **Runtime > Prompt:** Determinism is enforced by code (FSM), not just by "wishing" in the prompt.
  - **Absorbing States:** Once the agent enters FINISHED or ERROR, it stays there. No "ghost" iterations.
  - **Circuit-Breakers:** Loop detection acts as a deterministic halt for hallucination cycles.
  - **Instruction Salience:** Placing grounding data (History/Rules) at the generation point (prompt bottom) improves compliance in small models.

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


### Day 2 – Multi-Iteration Loop & Tool Integration

**Objectives Completed:**

- Refactored loop into `loop.py` with clear separation from `main.py`; main now acts as a lightweight facade.  
- Verified OutputParser works end-to-end with structured THOUGHT, ACTION, FINAL parsing, including error handling for malformed or missing fields.  
- Connected `search_local_knowledge` tool; simulated observation correctly stored in AgentState.  
- Tested minimal prompt scenarios to validate loop, parser, and tool interactions without excessive delays.

**Key Lessons:**

- LLMs still attempt to shortcut to FINAL; iteration-based guardrails are essential for consistent ReAct behavior.  
- Prompt complexity and model size directly affect local generation time; even first iteration can be slow with long prompts.  
- AgentState updates (thoughts, actions, observations) must be deterministic and explicitly handled per iteration.  
- Minimal prompts confirm structural correctness of loop and parser before introducing multi-step reasoning with real questions.

### Day 3 – Deterministic FSM & Strict JSON Contract

**Objectives Completed:**

- **Deterministic FSM Migration:** Replaced the fragile boolean-based `terminated` flag with a formal state machine (`RUNNING`, `FINISHED`, `ERROR`, `MAX_ITER`).
- **Strict JSON Output Contract:** Enforced a zero-tolerance JSON schema for LLM outputs, eliminating regex-based "fuzzy" parsing.
- **Cycle Detection (Loop Prevention):** Implemented runtime tracking of tool-call signatures to detect and terminate on "hallucination loops."
- **Prompt Recency Optimization:** Reordered prompt structure to place **Conversation History** and **Stopping Criteria** at the very bottom, leveraging the LLM's recency bias for faster convergence.
- **Refined Termination Logic:** Adjusted state transitions to allow successful finalization even on the final leg.

**Key Lessons:**

- **Absorbing States:** Formalizing terminal states ensures that once the agent leaves `RUNNING`, no further mutations or "ghost iterations" can occur.
- **JSON as a Protocol:** Treating the LLM as a structured data provider rather than a text generator significantly increases reliability in small (3B) models.
- **Hallucination Loop Countermeasures:** Autonomous agents are prone to repetitive reasoning cycles; runtime cycle detection is a mandatory "deterministic circuit-breaker" for production code.
- **Instruction Salience:** Local LLMs have limited attention; placing critical grounding data (History/Rules) at the end of the prompt (the "generation point") dramatically improves instruction following.
- **Pure Function Core:** By moving state mutation into a pure `transition()` function, the agent's logic becomes fully testable and decoupled from the non-deterministic LLM client.


### Day 4 – Reflection Stage & FAISS Vector Memory

**Objectives Completed:**

- **FSM Extended to 6 States:** `THINK → ACT → OBSERVE → REFLECT → DECIDE → THINK | STOP`. Reflection is an explicit FSM state, not embedded inside THINK.
- **Reflection/Self-Critique:** Added `agent/reflection.py` with `strict_reflection_parse()` enforcing `{"critique", "decision", "store_memory"}` JSON contract. LLM critiques each reasoning step before deciding to continue, search again, or stop.
- **FAISS Vector Memory:** Implemented `memory/vector_store.py` (`VectorMemory`) with `IndexFlatL2` and parallel `documents[]` list. Supports `add()` with deduplication and `search()` with `k=3`.
- **Local Embedding Pipeline:** `memory/embedding.py` wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim). No remote APIs.
- **Memory Persistence:** FAISS index + documents list saved to `data/memory.faiss` and `data/memory.json` after every write. Loaded at init if files exist — true long-term memory across runs.
- **Memory Retrieval Injection:** Before each THINK step, the research question + latest thought are embedded and top-3 memories are injected into the prompt context.
- **Memory Write Policy:** LLM proposes `store_memory: true/false`; the loop enforces deterministically. Error observations and duplicates are never stored.
- **Observation-Level Duplicate Detection:** Tools returning the same observation as a previous call trigger a `[DUPLICATE]` warning, nudging the LLM to change approach.
- **Dual-Phase FSM Tracking:** `fsm_phase` (THINK/ACT/OBSERVE/REFLECT/DECIDE) tracks position within an iteration, orthogonal to `status` (RUNNING/FINISHED/ERROR/MAX_ITER) which controls loop termination.

**Key Lessons:**

- **Recency Bias is Real:** Small LLMs pay most attention to the end of the prompt. Critical context (question, known facts, stopping criteria) must be at the bottom, not the top. Format examples and rules go at the top as reference material.
- **Memory Without Persistence is Pointless:** An in-memory-only vector store provides no value across runs. FAISS `write_index`/`read_index` + JSON document persistence is the minimum viable approach.
- **Observation-Level Guards > Input-Level Guards:** Tool-input loop detection misses cases where different inputs produce identical outputs (e.g., different substrings matching the same KB entry). Checking observation equality catches this.
- **FAISS Index Boundary Gotcha:** FAISS uses `-1` as a sentinel for "no more results" when `k > n_docs`. Python negative indexing silently wraps around instead of erroring — always guard with `idx >= 0`.
- **Reflection Needs Separation:** Making reflection an explicit FSM state (not embedded in THINK) keeps the JSON contracts clean and makes each phase independently testable.
- **Type Assumptions Kill:** When multiple tools return different types (str vs int vs dict), any code touching observations must handle polymorphic returns. The `Tool.run()` contract says `-> str` but implementations don't always comply.
- **Prompt Engineering for Small Models is Architecture:** For 3B models, prompt wording changes alone often fail. Effective "prompt engineering" means restructuring *what information appears where* and adding *architectural guardrails* (observation dedup, memory dedup) rather than hoping the model follows instructions.

---

## Daily Cheat Sheets

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
  
  
**Day 2 Cheat Sheet:**
```mermaid
flowchart TD
    Q[Input Question / Topic] --> S[AgentState Initialized]
    S --> P[Build Prompt with THOUGHT + Memory]
    P --> L[LLM Call via OllamaClient.generate]
    L --> R[Raw LLM Output]
    R -->|Parse| O[OutputParser: THOUGHT / ACTION / FINAL / ERROR]
    O --> C{Premature FINAL?}
    C -->|Yes, Iteration 0| I[Reject FINAL, Increment Iteration]
    C -->|No| U[Update AgentState: Add Thought, Action, Observation]
    U --> T{Is FINAL Present?}
    T -->|Yes| F[Terminate Agent, Set Final Answer]
    T -->|No| N[Increment Iteration, Continue Loop]
    F --> E[Return AgentState with Final Answer]
    N --> P
```

**Day 3 Cheat Sheet (Decoupled Pipeline):**
```mermaid
flowchart LR
    subgraph Input
        S[AgentState] --> P[Prompt Builder]
    end
    
    subgraph Generation
        P --> LLM[Ollama Client]
    end

    subgraph FSM_Logic
        LLM --> Parser[JSON Parser]
        Parser -->|Validated Dict| Transition[FSM Transition]
    end

    subgraph Side_Effects
        Transition -->|RUNNING + Tool| Tool[Tool Execution]
        Tool --> Inject[Observation Injection]
        Inject --> S
    end

    Transition -->|FINISHED/ERROR/MAX_ITER| Exit[Terminal State]
```

**Day 3 State Transitions:**
```mermaid
stateDiagram-v2
    [*] --> RUNNING: Question Received
    
    state RUNNING {
        direction lr
        T: transition()
        I: inject_observation()
        T --> I: action_type == 'tool'
        I --> T: Loop continues
    }
    
    RUNNING --> FINISHED: transition() -> status changed
    RUNNING --> ERROR: Parse Error / Loop Detected / Unknown Tool
    RUNNING --> MAX_ITER: iteration >= max_iterations
    
    FINISHED --> [*]
    ERROR --> [*]
    MAX_ITER --> [*]
    
    note right of RUNNING
       Guards:
       1. Strict JSON Parse
       2. Cycle Detection
       3. Schema Validation
    end note
```


**Day 4 Cheat Sheet (Full FSM with Reflection & Memory):**
```mermaid
flowchart TD
    Q[Research Question] --> E[Embed Question + Latest Thought]
    E --> MR[FAISS Search: Top-3 Memories]
    MR --> TP[Build Think Prompt with Memories]
    
    TP --> THINK[THINK: LLM Generates Action]
    THINK --> PARSE[Strict JSON Parse]
    PARSE --> TR[transition: Validate & Mutate State]
    
    TR -->|final_answer| DONE[FINISHED]
    TR -->|tool action| ACT[ACT: Execute Tool]
    
    ACT --> DUP{Duplicate Observation?}
    DUP -->|Yes| WARN["[DUPLICATE] Warning Injected"]
    DUP -->|No| OBS[OBSERVE: Store Result]
    WARN --> OBS
    
    OBS --> REFLECT[REFLECT: LLM Critiques Step]
    REFLECT --> RPARSE[Strict Reflection Parse]
    RPARSE --> DECIDE[DECIDE: reflect_transition]
    
    DECIDE -->|stop| SYNTH[Synthesize Final Answer]
    DECIDE -->|continue / search_again| MW{store_memory?}
    
    MW -->|true + not error + not duplicate| STORE[FAISS Add + Disk Save]
    MW -->|false| SKIP[Skip Memory Write]
    
    STORE --> E
    SKIP --> E
    SYNTH --> DONE
```

**Day 4 State Transitions:**
```mermaid
stateDiagram-v2
    [*] --> THINK: Question Received

    THINK --> ACT: action_type == tool
    THINK --> FINISHED: action_type == final_answer
    THINK --> ERROR: Parse Error / Loop Detected
    THINK --> MAX_ITER: iteration >= max

    ACT --> OBSERVE: Tool Executed
    OBSERVE --> REFLECT: Observation Stored

    REFLECT --> DECIDE: Critique Parsed
    REFLECT --> ERROR: Reflection Parse Error

    DECIDE --> THINK: continue / search_again
    DECIDE --> FINISHED: stop

    FINISHED --> [*]
    ERROR --> [*]
    MAX_ITER --> [*]

    note right of REFLECT
       Reflection JSON:
       critique, decision,
       store_memory
    end note

    note left of DECIDE
       Memory Write:
       Only if store_memory=true
       AND not error obs
       AND not duplicate
    end note
```
---
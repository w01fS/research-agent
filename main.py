from agent.loop import AgentLoop
from agent.state import AgentState
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from tools.search_local import SearchLocalKnowledgeTool
from agent.llm import OllamaClient


def main():
    # 1. Initialize tool registry
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchLocalKnowledgeTool())

    # 2. Initialize LLM client
    llm = OllamaClient(model="llama3.2:3b")

    # 3. Initialize agent state
    question = "what are the roadblocks to semiconductor supply chain?"
    # question = "what is 25 + 65?"
    state = AgentState(question=question)

    # 4. Create agent loop
    agent = AgentLoop(llm_client=llm, registry=registry, max_iterations=6)

    # 5. Run the agent (returns final AgentState)
    state = agent.run(state=state)

    # 6. Report result based on FSM status
    print("\n========================")
    if state.status == "FINISHED":
        print("FINAL ANSWER:")
        print(state.final_answer)
    elif state.status == "ERROR":
        print(f"AGENT ERROR: {state.error_reason}")
    elif state.status == "MAX_ITER":
        print(f"MAX ITERATIONS: {state.error_reason}")
    else:
        print(f"UNEXPECTED STATUS: {state.status}")
    print("========================")


if __name__ == "__main__":
    main()
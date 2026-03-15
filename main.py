from agent.loop import AgentLoop
from agent.state import AgentState
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from tools.search_local import SearchLocalKnowledgeTool
from agent.llm import OllamaClient
from memory.embedding import EmbeddingModel
from memory.vector_store import VectorMemory


def main():
    # 1. Initialize tool registry
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchLocalKnowledgeTool())

    # 2. Initialize LLM client
    llm = OllamaClient(model="llama3.2:3b")

    # 3. Initialize embedding model and vector memory
    print("Loading embedding model...")
    embedder = EmbeddingModel()
    memory = VectorMemory(dim=384, storage_dir="data")
    print("Embedding model ready.")

    # 4. Initialize agent state
    # question = "what are the dependencies of semiconductor supply chain?"
    question = "what is 25 + 65?"
    state = AgentState(question=question)

    # 5. Create agent loop
    agent = AgentLoop(
        llm_client=llm,
        registry=registry,
        embedder=embedder,
        memory=memory,
        max_iterations=6,
    )

    # 6. Run the agent (returns final AgentState)
    state = agent.run(state=state)

    # 7. Report result based on FSM status
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

    # Summary stats
    print(f"\nIterations: {state.iteration}")
    print(f"Reflections: {len(state.reflections)}")
    print(f"Memories stored: {memory.index.ntotal}")
    print("========================")


if __name__ == "__main__":
    main()
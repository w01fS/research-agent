from agent.loop import AgentLoop
from agent.state import AgentState
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from tools.search_local import SearchLocalKnowledgeTool
from agent.llm import OllamaClient  # new client

def main():
    # 1️⃣ Initialize tool registry
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchLocalKnowledgeTool())

    # 2️⃣ Initialize LLM client
    llm = OllamaClient(model="llama3.2:3b")

    # 3️⃣ Initialize agent state
    question = "What is 2 + 2?"
    state = AgentState(question=question)

    # 4️⃣ Create agent loop
    agent = AgentLoop(llm_client=llm, registry=registry, max_iterations=6)

    # 5️⃣ Run the agent
    final_answer = agent.run(initial_prompt=question, state=state)

    print("\n========================")
    print("FINAL ANSWER:")
    print(final_answer)
    print("========================")


if __name__ == "__main__":
    main()
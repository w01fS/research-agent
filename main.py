from agent.state import AgentState
from agent.prompt import build_prompt
from agent.llm import call_llm
from agent.parser import parse_llm_output, ParseError

def run_agent(question: str):
    state = AgentState(question=question)

    while not state.terminated:
        if state.iteration >= state.max_iterations:
            print("Max iterations reached.")
            break

        prompt = build_prompt(state)
        output = call_llm(prompt)

        print("\nRAW LLM OUTPUT:\n", output)

        try:
            parsed = parse_llm_output(output)
        except ParseError as e:
            print("Parse error:", e)
            state.iteration += 1
            continue

        if parsed["final"] and state.iteration == 0:
            print("Rejecting premature FINAL.")
            state.iteration += 1
            continue

        state.thoughts.append(parsed["thought"])
        state.actions.append(parsed["action"] or "")
        state.observations.append("Simulated observation")

        if parsed["final"]:
            state.final_answer = parsed["final"]
            state.terminated = True

        state.iteration += 1

    return state


if __name__ == "__main__":
    result = run_agent("Explain the long-term impact of semiconductor supply chains.")
    print("\nFINAL ANSWER:\n", result.final_answer)
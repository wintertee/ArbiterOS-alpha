import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from rich.logging import RichHandler

import arbiteros_alpha.instructions as Instr
from arbiteros_alpha import ArbiterOSAlpha, print_history
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter

logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.DEBUG,
    handlers=[RichHandler()],
)

# 1. Setup OS

os = ArbiterOSAlpha(backend="langgraph")

# Policy: Prevent direct generate->toolcall without proper flow
history_checker = HistoryPolicyChecker(
    name="no_direct_toolcall",
    bad_sequence=[Instr.GENERATE, Instr.TOOL_CALL],
)

# Policy: If confidence is low, regenerate the response
confidence_router = MetricThresholdPolicyRouter(
    name="regenerate_on_low_confidence",
    key="confidence",
    threshold=0.6,
    target="generate",
)


# if add this checker, intended error will be raised
os.add_policy_checker(history_checker)
os.add_policy_router(confidence_router)

# 2. Langgraph stuff


class State(TypedDict):
    """State for a simple AI assistant with tool usage and self-evaluation."""

    query: str
    response: str
    tool_result: str
    confidence: float


@os.instruction(Instr.GENERATE)
def generate(state: State) -> State:
    """Generate a response to the user query."""

    # Check if this is a retry (response already exists)
    is_retry = bool(state.get("response"))

    if is_retry:
        # On retry, generate a longer, better response
        response = "Here is my comprehensive and detailed response with much more content and explanation."
    else:
        # First attempt: short response (will have low confidence)
        response = "Short reply."

    return {"response": response}


@os.instruction(Instr.TOOL_CALL)
def tool_call(state: State) -> State:
    """Call external tools to enhance the response."""
    return {"tool_result": "ok"}


@os.instruction(Instr.EVALUATE_PROGRESS)
def evaluate(state: State) -> State:
    """Evaluate confidence in the response quality."""
    # Heuristic: response quality based on length
    # Short response (<60 chars) = low confidence (<0.6)
    # Longer response (>=60 chars) = high confidence (>=0.6)
    response_length = len(state["response"])
    confidence = min(response_length / 100.0, 1.0)
    return {"confidence": confidence}


builder = StateGraph(State)
builder.add_node(generate)
builder.add_node(tool_call)
builder.add_node(evaluate)

builder.add_edge(START, "generate")
builder.add_edge("generate", "tool_call")
builder.add_edge("tool_call", "evaluate")
builder.add_edge("evaluate", END)

graph = builder.compile()

# 3. Run graph


def main():
    initial_state: State = {
        "query": "What is AI?",
        "response": "",
        "tool_result": "",
        "confidence": 0.0,
    }
    for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
        logger.info(f"Current state: {chunk}\n")


if __name__ == "__main__":
    main()
    print_history(os.history)

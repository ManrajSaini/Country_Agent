import logging
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.intent import intent_node
from agent.nodes.synthesize import synthesize_node
from agent.nodes.tool_call import tool_node

logger = logging.getLogger(__name__)

def route_after_intent(state: AgentState) -> str:
    """
    Conditional routing function after intent node.
    If is_valid is True, proceed to tool node else skip to synthesize node.
    """
    if state.get("is_valid"):
        logger.debug("Intent valid - routing to tool node")
        return "tool_node"
    else:
        logger.debug("Intent invalid - short-circuit to synthesize node")
        return "synthesize_node"
    
def build_graph() -> StateGraph:
    """
    Build and compile the country info agent graph.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_node", intent_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("synthesize_node", synthesize_node)

    workflow.set_entry_point("intent_node")

    workflow.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "tool_node": "tool_node",
            "synthesize_node": "synthesize_node"
        }
    )

    workflow.add_edge("tool_node", "synthesize_node")
    workflow.add_edge("synthesize_node", END)

    return workflow.compile()

graph = build_graph()
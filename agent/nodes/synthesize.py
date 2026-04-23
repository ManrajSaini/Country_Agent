import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from config import config

logger = logging.getLogger(__name__)

SUCCESS_SYSTEM_PROMPT = """
You are a helpful country information assistant.
You will be given structured data about a country and the user's original question.
Your job is to answer the question naturally and concisely based STRICTLY on the data provided.
Do not add any information not present in the data.
Do not make up facts. If a field is None, say that information is not available.
"""

ERROR_SYSTEM_PROMPT = """
You are a helpful country information assistant.
You were unable to retrieve data for the user's request.
Respond politely and helpfully explaining what went wrong.
Keep it concise with one or two sentences maximum.
"""


def synthesize_node(state: AgentState) -> dict:
    """
    Synthesizes a natural language answer from extracted country data or error.
    Returns only the keys it modifies in state.
    """
    question = state["question"]
    country_name = state.get("country_name")
    raw_country_data = state.get("raw_country_data")    # get method for optional fields
    tool_error = state.get("tool_error")

    logger.info(f"Synthesize node processing - country: '{country_name}', error: {tool_error}")

    llm = ChatGroq(
        model=config.model_name,
        api_key=config.llm_api_key,
        temperature=0,
    )

    if tool_error:
        messages = [
            SystemMessage(content=ERROR_SYSTEM_PROMPT),
            HumanMessage(content=f"User asked: {question}\n\nError: {tool_error}")
        ]
    else:
        human_content = f"""
User question: {question}

Country: {country_name}
Data: {raw_country_data}

Answer the user's question naturally based strictly on the data above.
"""
        messages = [
            SystemMessage(content=SUCCESS_SYSTEM_PROMPT),
            HumanMessage(content=human_content)
        ]

    try:
        response = llm.invoke(messages)
        final_answer = response.content.strip()
        logger.info(f"Synthesize node produced answer: '{final_answer}'")

        return {"final_answer": final_answer}

    except Exception as e:
        logger.error(f"Synthesize node failed unexpectedly: {e}")
        return {"final_answer": "I'm sorry, I was unable to generate an answer at this time. Please try again."}
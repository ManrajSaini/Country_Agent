import json
import logging
import httpx
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import config
from agent.tools.countries_api import FIELD_MAP
from agent.state import AgentState

logger = logging.getLogger(__name__)
http_client = httpx.Client(verify=not config.disable_ssl_verify)

ALLOWED_FIELDS = list(FIELD_MAP.keys())

SYSTEM_PROMPT = """
You are an intent extraction assistant. Your job is to analyze a user question about countries and extract structured information.

You must respond with ONLY a valid JSON object — no explanation, no preamble, no markdown.

The JSON must have exactly these three fields:
- "country_name": the name of the country mentioned (string), or null if none found
- "requested_fields": a list of fields the user is asking about, chosen ONLY from this allowed list: {allowed_fields}
- "is_valid": true if you found a country name AND at least one valid field, false otherwise

Examples:

Question: "What is the population of Germany?"
{{"country_name": "Germany", "requested_fields": ["population"], "is_valid": true}}

Question: "What is the capital and population of Brazil?"
{{"country_name": "Brazil", "requested_fields": ["capital", "population"], "is_valid": true}}

Question: "Tell me about dogs"
{{"country_name": null, "requested_fields": [], "is_valid": false}}

Question: "What is the population of Narnia?"
{{"country_name": "Narnia", "requested_fields": ["population"], "is_valid": true}}
""".format(allowed_fields=ALLOWED_FIELDS)


def intent_node(state: AgentState) -> dict:
    """
    Extract country name, requested fields and validity from user question.
    Return only the keys it modifies.
    """
    question = state["question"]
    logger.info(f"Intent node received question: '{question}'")

    llm = ChatGroq(
        model=config.model_name,
        api_key=config.llm_api_key,
        temperature=0,  # dont be creative and give deterministic O/P
        http_client=http_client
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    try:
        response = llm.invoke(messages)
        raw_content = response.content.strip()
        logger.debug(f"LLm raw response: {raw_content}")

        parsed = json.loads(raw_content)

        country_name = parsed.get("country_name")
        requested_fields = parsed.get("requested_fields", [])
        is_valid = parsed.get("is_valid", False)

        if is_valid and (not country_name or not requested_fields):
            logger.warning("LLm returned is_valid=True but missing country or fields")
            is_valid = False

        valid_fields = [f for f in requested_fields if f in FIELD_MAP]
        if len(valid_fields) != len(requested_fields):
            logger.warning(f"Some fields were filtered out. Original: {requested_fields}, VAlid: {valid_fields}")

        logger.info(f"Intent extracted - country: '{country_name}', fields: {valid_fields}, valid: {is_valid}")

        return {
            "country_name": country_name,
            "requested_fields": valid_fields,
            "is_valid": is_valid
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLm response as JSON: {e}")
        return {
            "country_name": None,
            "requested_fields": [],
            "is_valid": False
        }
    
    except Exception as e:
        logger.error(f"Intent node failed unexpectedly: {e}")
        return {
            "country_name": None,
            "requested_fields": [],
            "is_valid": False
        }
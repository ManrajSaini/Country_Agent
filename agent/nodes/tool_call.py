import logging
from agent.state import AgentState
from agent.tools.countries_api import fetch_country_data, extract_fields

logger = logging.getLogger(__name__)

def tool_node(state: AgentState) -> dict:
    """
    Fetches country data from REST countries API and extracts requested fields.
    Returns only keys it modified in state.
    """
    country_name = state["country_name"]
    requested_fields = state["requested_fields"]

    logger.info(f"Tool node fetching data for country: '{country_name}', fields: {requested_fields}")

    raw_data, error = fetch_country_data(country_name)

    if error:
        logger.warning(f"Tool node fetch failed: {error}")
        return {
            "raw_country_data": None,
            "tool_error": error,
        }
    
    extracted = extract_fields(raw_data, requested_fields)
    logger.info(f"Tool node extracted fields: {extracted}")

    return {
        "raw_country_data": extracted,
        "tool_error": None
    }
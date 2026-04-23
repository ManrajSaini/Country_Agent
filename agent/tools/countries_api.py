import requests
import logging
from typing import Callable
from config import config

logger = logging.getLogger(__name__)

# keys must match exactly what the intent node puts in requested fields
FIELD_MAP: dict[str, Callable[[dict], object]] = {
    "population": lambda c: c.get("population"),
    "capital": lambda c: c.get("capital", [None])[0],
    "languages": lambda c: list(c.get("languages", {}).values()),
    "area": lambda c: c.get("area"),
    "region": lambda c: c.get("region"),
    "subregion": lambda c: c.get("subregion"),
    "flag": lambda c: c.get("flags", {}),
    "borders": lambda c: c.get("borders", []),
    "timezones": lambda c: c.get("timezones", []),
    "currencies": lambda c: {
        code: info.get("name")
        for code, info in c.get("currencies", {}).items()
    },
}

def fetch_country_data(country_name: str) -> tuple[dict | None, str | None]:
    """
    Fetches raw country data from REST countries API
    Returns (data, error)
    """
    url = config.api_base_url.format(country=country_name)
    try:
        response = requests.get(url, timeout=config.request_timeout)
        
        if response.status_code == 404:
            return None, f"Country '{country_name}' not found."
        
        response.raise_for_status()

        data = response.json()
        return data[0], None
    
    except requests.Timeout:
        return None, f"Request timed out fetching data for '{country_name}'."
    except requests.RequestException as e:
        return None, f"API request failed: {str(e)}"
    
def extract_fields(raw_data: dict, requested_fields: list[str]) -> dict:
    """
    Extract only the requested fields from raw API response
    using FIELD_MAP, missing fields are stored as None.
    """
    extracted = {}
    for field in requested_fields:
        if field not in FIELD_MAP:
            logger.warning(f"Unknown field requested: {field}")
            extracted[field] = None
            continue

        try:
            extracted[field] = FIELD_MAP[field](raw_data)
        except Exception as e:
            logger.warning(f"Failed to extract field '{field}' : {e}")
            extracted[field] = None
        
    return extracted
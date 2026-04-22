from typing import TypedDict

class AgentState(TypedDict, total=False):
    question: str   # original user input
    country_name: str | None  # extracted by intent node
    requested_fields: list[str] | None # ["population", "capital"]
    is_valid: bool  # False means short circuit before API call
    raw_country_data: dict | None # only requested fields extracted from API
    tool_error: str | None    # error reason if API failed
    final_answer: str | None  # from synthesis node
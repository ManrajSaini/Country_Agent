from dataclasses import dataclass
from typing import Literal
import os

LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

@dataclass
class Config:
    model_name: str
    llm_api_key: str
    api_base_url: str
    request_timeout: float
    max_retries: int
    log_level: LOG_LEVELS


config = Config(
    model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant-on_demand"),
    llm_api_key = os.getenv("LLM_API_KEY", ""),
    api_base_url = "https://restcountries.com/v3.1/name/{country}",
    request_timeout = 10.0,
    max_retries = 3,
    log_level = "INFO"
)

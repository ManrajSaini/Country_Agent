import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()   # load env vars before graph use them

from agent.graph import graph


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Question must not be empty")
        return v.strip()
    
class AnswerResponse(BaseModel):
    answer: str
    country: str | None = None
    fields_requested: list[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Country information agent starting up...")
    yield
    logger.info("Country information agent shutting down...")


app = FastAPI(
    title="Country information agent",
    description="An AI agent that answers questions about countries using LangGraph.",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    logger.info(f"Received question: '{request.question}'")

    try:
        result = graph.invoke({
            "question": request.question,
            "country_name": None,
            "requested_fields": [],
            "is_valid": False,
            "raw_country_data": None,
            "tool_error": None,
            "final_answer": None,
        })

        final_answer = result.get("final_answer")
        if not final_answer:
            raise HTTPException(status_code=500, detail="Agent failed to produce an answer.")
        
        logger.info(f"Answer produced for '{request.question}': '{final_answer}'")

        return AnswerResponse(
            answer=final_answer,
            country=result.get("country_name"),
            fields_requested=result.get("requested_fields", [])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again")
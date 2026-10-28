from datetime import datetime
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    message: str = Field(min_length=3, max_length=2000)
    make: str = Field(min_length=1, max_length=60)
    model: str = Field(min_length=1, max_length=60)
    year: int = Field(ge=1980, le=2030)
    conversation_id: UUID | None = None


class LabelScore(BaseModel):
    category: str
    score: float = Field(ge=0, le=1)


class Classification(BaseModel):
    category: str  # Primary predicted group for legacy frontend compatibility.
    confidence: float | None = None  # Model score, not calibrated probability.
    method: str
    labels: list[LabelScore] = Field(default_factory=list)  # Multi-label predictions.


class Source(BaseModel):
    title: str
    url: str
    source_id: str
    excerpt: str


class AnalyzeResponse(BaseModel):
    conversation_id: str
    answer: str
    classification: Classification
    sources: list[Source]
    recall_status: Literal['ok', 'unavailable', 'missing_vehicle_details', 'not_queried']
    generation_mode: Literal['llm', 'evidence_only', 'llm_unavailable', 'intent_guardrail']
    generation_error: str | None = None
    intent: Literal['automotive', 'out_of_scope', 'clarification'] = 'automotive'


class Chat(BaseModel):
    id: str
    title: str
    created_at: str


class ChatMessage(BaseModel):
    id: str
    chat_id: str
    role: Literal['user', 'assistant']
    content: str
    created_at: str
    metadata: dict = Field(default_factory=dict)

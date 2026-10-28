"""Separate local-only AG-UI server, preserving /api/analyze and current auth.

Run: uvicorn backend.app.copilot_server:app --host 127.0.0.1 --port 8123
Do not expose publicly: this initial CopilotKit prototype has no user auth.
"""
from fastapi import FastAPI
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from agent.story_agent import story_graph

app = FastAPI(title='AutoAssist CopilotKit agent (local development only)')

add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAGUIAgent(
        name='autoassist',
        description='Vehicle-story intake, DistilBERT symptom classification and NHTSA-grounded explanations.',
        graph=story_graph,
    ),
    path='/agent',
)

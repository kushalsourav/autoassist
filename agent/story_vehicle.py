"""Legacy import compatibility. Vehicle identity is now LLM-extracted.

Vehicle normalization/validation lives in agent.agentic_planner.VehicleIdentity.
No fixed manufacturer list or symptom-keyword routing is used by story_agent.
"""
from agent.agentic_planner import VehicleIdentity, merge_vehicle

__all__ = ['VehicleIdentity', 'merge_vehicle']

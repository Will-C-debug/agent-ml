"""A local machine-learning engineer agent built on LangChain, LangGraph and FLAML AutoML."""

from ml_agent.agent import build_agent
from ml_agent.config import Settings

__all__ = ["Settings", "build_agent"]
__version__ = "0.1.0"

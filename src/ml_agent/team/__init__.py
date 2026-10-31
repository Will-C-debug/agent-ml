"""A team of specialist agents that searches for the best scikit-learn model.

Planner → Data → Analysis → Modeling → Review (loops back on "revise") → final evaluation →
Visualization → export, orchestrated as a LangGraph ``StateGraph``. See ``graph.py``.
"""

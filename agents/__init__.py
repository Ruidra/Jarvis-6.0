"""
agents/ — JARVIS Specialized Agents.

Each agent is a domain specialist that receives a task, performs the work
using available tools/LLM, reviews its own output, and returns a structured
result to the manager (JARVIS).

Available agents:
  - web_agent      : websites, landing pages, web apps, HTML/CSS/JS
  - photo_agent    : image generation, editing, analysis, design
  - video_agent    : video scripts, editing plans, thumbnail ideas
  - app_agent      : desktop/mobile apps, installers, packaging
  - code_agent     : code review, debugging, refactoring, implementation
  - research_agent : deep research, reports, fact-checking
  - data_agent     : data analysis, CSV/Excel, databases, visualization
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "web_agent",
    "photo_agent",
    "video_agent",
    "app_agent",
    "code_agent",
    "research_agent",
    "data_agent",
]

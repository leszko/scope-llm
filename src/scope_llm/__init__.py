"""Local LLM plugin for Daydream Scope."""

from .plugin import LLMPlugin

plugin = LLMPlugin()

__all__ = ["plugin", "LLMPlugin"]

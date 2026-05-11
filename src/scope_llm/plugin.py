"""Plugin class wiring LLMNode into the Scope node registry."""

import logging

from scope.core.plugins import hookimpl

from .node import LLMNode

logger = logging.getLogger(__name__)


class LLMPlugin:
    """Registers the Local LLM node."""

    @hookimpl
    def register_nodes(self, register):
        register(LLMNode)
        logger.info("Registered Local LLM node")

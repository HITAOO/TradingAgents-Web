"""Shared helpers for invoking an agent with structured output and a graceful fallback.

Pattern:
1. At agent creation, wrap the LLM with with_structured_output(Schema).
   If the provider does not support it, the wrap is skipped.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call fails for any reason, fall back to a
   plain llm.invoke so the pipeline never blocks.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return llm.with_structured_output(schema) or None if unsupported."""
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run structured call and render to markdown; fall back to free-text on failure."""
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content

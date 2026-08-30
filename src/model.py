"""Orchestrates one full prediction: prompt -> FunctionCallResult.

This is where functions_definition.json meets the constrained decoder:
it builds the prompts sent to the LLM, picks which decoder function to
use for each parameter type, and assembles the final typed result.
"""

import re

from .decoder import (
    generate_constrained_boolean,
    generate_constrained_choice,
    generate_constrained_number,
    generate_constrained_string,
)
from .llm import LLMWrapper
from .models import FunctionCallResult, FunctionDefinition

# A small, fixed vocabulary mapping common "category words" to a regex.
# This only covers the handful of categories a 0.6B model is unlikely to
# spell out correctly on its own; it is not meant to grow indefinitely.
CATEGORY_REGEX = {
    "numbers": r"\d+",
    "digits": r"\d+",
    "vowels": r"[aeiouAEIOU]",
    "letters": r"[A-Za-z]",
    "whitespace": r"\s",
    "spaces": r"\s",
}

QUOTED_SPAN_PATTERN = re.compile(r"'([^']+)'|\"([^\"]+)\"")


def _quoted_spans(prompt: str) -> list[str]:
    """Return every single- or double-quoted span in the prompt, in order."""
    spans = []
    for single, double in QUOTED_SPAN_PATTERN.findall(prompt):
        spans.append(single if single else double)
    return spans


def _deterministic_string(
    prompt: str,
    param_name: str,
    already_extracted: dict[str, object],
) -> str | None:
    """Best-effort, rule-based extraction for a value already explicitly
    present in the prompt (a quoted span, a trailing word, a category
    keyword...). Returns None when nothing safe is found, so the caller
    can fall back to constrained LLM generation instead of guessing.

    This exists because forcing a 0.6B model to freely copy a value that
    is already written verbatim in the request is unreliable -- it is
    both unnecessary and error-prone.
    """
    used = {str(value) for value in already_extracted.values()}
    spans = [span for span in _quoted_spans(prompt) if span not in used]

    if param_name in ("source_string", "s", "text", "string"):
        return max(spans, key=len) if spans else None

    if param_name in ("regex", "pattern"):
        if spans:
            return spans[0]
        lowered = prompt.lower()
        for keyword, pattern in CATEGORY_REGEX.items():
            if keyword in lowered:
                return pattern
        return None

    if param_name in ("replacement", "value"):
        if spans:
            return spans[0]
        match = re.search(r"\bwith\s+([A-Za-z0-9_]+)\s*[.?!]?\s*$", prompt)
        return match.group(1) if match else None

    if param_name == "name":
        words = re.findall(r"[A-Za-z']+", prompt)
        return words[-1] if words else None

    return None


def _choose_function(
    llm: LLMWrapper, prompt: str, functions: list[FunctionDefinition]
) -> FunctionDefinition:
    """Ask the LLM to pick which function matches the prompt."""
    names = [f.name for f in functions]
    descriptions = "\n".join(f"- {f.name}: {f.description}" for f in functions)

    context = (
        "You must choose which function to call to satisfy the user's request.\n"
        f"Available functions:\n{descriptions}\n"
        f"User request: {prompt}\n"
        "Function to call: "
    )
    input_ids = llm.encode(context)
    chosen_name = generate_constrained_choice(llm, input_ids, names)

    for function in functions:
        if function.name == chosen_name:
            return function

    raise ValueError(f"LLM chose an unknown function: {chosen_name!r}")


def _extract_parameter(
    llm: LLMWrapper,
    prompt: str,
    function: FunctionDefinition,
    param_name: str,
    param_type: str,
    already_extracted: dict[str, object],
) -> object:
    """Extract one typed parameter value from the prompt.

    For strings, an explicit value already present in the prompt is
    extracted deterministically first (see _deterministic_string).
    The LLM is only used as a fallback, or for types that are not
    plain copies of prompt text (numbers, booleans).

    already_extracted holds the parameters already filled in for this
    same function call, so neither the deterministic pass nor the LLM
    blindly repeats a value already used for another parameter.
    """
    if param_type == "string":
        deterministic_value = _deterministic_string(prompt, param_name, already_extracted)
        if deterministic_value is not None:
            return deterministic_value

    already_text = (
        f"Already extracted parameters: {already_extracted}\n"
        if already_extracted
        else ""
    )
    context = (
        f"User request: {prompt}\n"
        f"Function being called: {function.name} ({function.description})\n"
        f"{already_text}"
        f"Extract the value of parameter '{param_name}' from the request.\n"
        f"Value: "
    )
    input_ids = llm.encode(context)

    if param_type == "number":
        return generate_constrained_number(llm, input_ids)
    if param_type == "boolean":
        return generate_constrained_boolean(llm, input_ids)
    if param_type == "string":
        return generate_constrained_string(llm, input_ids)

    raise ValueError(f"Unsupported parameter type: {param_type!r}")


def predict_function_call(
    llm: LLMWrapper, prompt: str, functions: list[FunctionDefinition]
) -> FunctionCallResult:
    """Run the full pipeline for a single prompt and return the result."""
    function = _choose_function(llm, prompt, functions)

    parameters: dict[str, object] = {}
    for param_name, param_schema in function.parameters.items():
        param_type = param_schema.get("type", "string")
        parameters[param_name] = _extract_parameter(
            llm, prompt, function, param_name, param_type, parameters
        )

    return FunctionCallResult(
        prompt=prompt,
        name=function.name,
        parameters=parameters,
    )

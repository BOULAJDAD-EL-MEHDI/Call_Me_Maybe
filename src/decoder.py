"""Constrained decoding primitives.

Instead of letting the LLM freely generate text (unreliable for a 0.6B
model), every function here restricts, at each generation step, which
tokens are allowed. Invalid tokens are simply never considered, so the
final result is guaranteed to have the right shape.

There are two kinds of constrained generation used in this project:

1. `generate_constrained_choice`: pick exactly one string from a known,
   finite list (used for function names and booleans).
2. `generate_constrained_number`: generate digits/'-'/'.' only, stopping
   as soon as the model would otherwise produce a non-numeric character.

Both work the same way: at every step we ask the SDK for logits over the
*entire* vocabulary, but we only ever pick the highest-scoring token
among the ones that keep the output valid. Invalid tokens are treated as
if their logit were -inf (we just never select them).
"""

import re

from .llm import LLMWrapper

MAX_VALUE_TOKENS = 20  # safety cap so a broken prompt can't loop forever
NUMBER_CHARS = set("0123456789.-")
NUMBER_PATTERN = re.compile(r"^-?\d*\.?\d*$")
COMPLETE_NUMBER_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def generate_constrained_choice(
    llm: LLMWrapper, input_ids: list[int], choices: list[str]
) -> str:
    """Generate tokens until the output exactly matches one of `choices`.

    At each step, only tokens that keep the generated text a valid
    prefix of at least one choice are considered.
    """
    generated = ""
    ids = list(input_ids)

    while generated not in choices:
        logits = llm.get_logits(ids)
        best_id = -1
        best_score = float("-inf")

        for token_id, score in enumerate(logits):
            if generated == "":
                token_str = llm.clean_token(token_id).strip(" ")
            else:
                token_str = llm.clean_token(token_id)
            if token_str == "":
                continue
            candidate = generated + token_str
            is_valid = any(choice.startswith(candidate) for choice in choices)
            if is_valid and score > best_score:
                best_score = score
                best_id = token_id

        if best_id == -1:
            raise ValueError(f"No valid token found among choices: {choices}")

        if generated == "":
            token_str = llm.clean_token(best_id).strip(" ")
        else:
            token_str = llm.clean_token(best_id)
        generated += token_str
        ids.append(best_id)

    return generated


def generate_constrained_number(llm: LLMWrapper, input_ids: list[int]) -> float:
    """Generate a JSON number, one token at a time.

    Only tokens made entirely of digits, '.', or '-' are ever considered.
    We stop as soon as we already have a complete, valid number AND the
    model's genuine (unconstrained) top choice is not numeric -- meaning
    it wants to move on (comma, space, end of sentence...). This avoids
    forcing more digits just because *some* digit scored highest among
    the numeric-only candidates.
    """
    generated = ""
    ids = list(input_ids)

    for _ in range(MAX_VALUE_TOKENS):
        logits = llm.get_logits(ids)

        # What would the model pick with NO constraint at all?
        unconstrained_best_id = max(range(len(logits)), key=lambda i: logits[i])
        unconstrained_best = llm.clean_token(unconstrained_best_id).strip()
        model_wants_more_digits = bool(unconstrained_best) and all(
            ch in NUMBER_CHARS for ch in unconstrained_best
        )

        if generated and COMPLETE_NUMBER_PATTERN.match(generated) and not model_wants_more_digits:
            break  # we already have a full number and the model wants to stop

        best_id = -1
        best_score = float("-inf")
        for token_id, score in enumerate(logits):
            token_str = llm.clean_token(token_id).strip()
            if not token_str or any(ch not in NUMBER_CHARS for ch in token_str):
                continue
            candidate = generated + token_str
            if NUMBER_PATTERN.match(candidate) and score > best_score:
                best_score = score
                best_id = token_id

        if best_id == -1:
            break  # no valid numeric token left -> the number is complete

        token_str = llm.clean_token(best_id).strip()
        generated += token_str
        ids.append(best_id)

    if not COMPLETE_NUMBER_PATTERN.match(generated):
        raise ValueError(f"Model failed to produce a valid number, got: {generated!r}")

    return float(generated)


def generate_constrained_string(llm: LLMWrapper, input_ids: list[int]) -> str:
    """Generate a JSON string's content (without the surrounding quotes).

    We stop as soon as the model wants to produce a quote OR a newline
    (BPE tokenizers like Qwen's represent a newline as the character
    'Ċ'). A newline means the model has finished its answer and is
    moving on to new commentary -- without this check, a small model
    tends to keep rambling well past the actual answer.
    We do not force any character set here: any text is valid string
    content, we just cap the length as a safety net.
    """
    generated = ""
    ids = list(input_ids)

    for _ in range(MAX_VALUE_TOKENS):
        logits = llm.get_logits(ids)
        best_id = max(range(len(logits)), key=lambda i: logits[i])
        token_str = llm.clean_token(best_id)

        if '"' in token_str or "\u010a" in token_str or "\n" in token_str:
            break

        generated += token_str
        ids.append(best_id)

    return generated.strip().strip("'\"")


def generate_constrained_boolean(llm: LLMWrapper, input_ids: list[int]) -> bool:
    """Generate a JSON boolean using the finite-choice decoder."""
    choice = generate_constrained_choice(llm, input_ids, ["true", "false"])
    return choice == "true"

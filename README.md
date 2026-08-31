*This project has been created as part of the 42 curriculum by eboulajd.*

# Call Me Maybe — Function Calling with Constrained Decoding

## Description

This project implements a function-calling pipeline that translates natural
language prompts (e.g. *"What is the sum of 2 and 3?"*) into structured,
schema-compliant function calls (e.g. `fn_add_numbers(a=2.0, b=3.0)`),
using a small local LLM (`Qwen/Qwen3-0.6B`).

Small language models are unreliable at spontaneously producing valid JSON.
Instead of prompting the model and hoping for well-formed output, this
project uses **constrained decoding**: at every generation step, the raw
logits returned by the model are filtered so that only tokens compatible
with the expected JSON schema can ever be chosen. This guarantees 100%
valid, schema-compliant output regardless of how confident (or not) the
model is.

## Instructions

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
make install   # uv sync
make run       # uv run python -m src
make debug     # run under pdb
make lint      # flake8 + mypy
make clean     # remove caches
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

**Environment note:** `llm_sdk` depends on `torch`, which can be large
(several GB with CUDA support). This project targets CPU execution;
if disk space is limited, configure `uv` to use the CPU-only `torch`
wheel and point `UV_CACHE_DIR` / `HF_HOME` to a partition with enough
free space.

## Algorithm Explanation

The pipeline works in two stages per prompt:

1. **Function selection** (`generate_constrained_choice` in `decoder.py`):
   the model is given the prompt and the list of available function names.
   At each generation step, only tokens that keep the generated text a
   valid prefix of one of the known function names are considered; every
   other token's logit is effectively treated as `-inf`. Generation stops
   as soon as the text exactly matches one full function name.

2. **Parameter extraction** (per declared type, also in `decoder.py`):
   - `number`: only tokens made of digits, `.`, or `-` are considered.
     Generation stops once a syntactically complete number has been built
     **and** the model's own unconstrained top choice is no longer a
     digit — meaning it wants to move on to something else (a comma, a
     space, etc.).
   - `string`: the model is allowed to pick freely (unconstrained
     top-1), but generation stops as soon as it wants to produce a quote
     or a newline, since that means the value is finished.
   - `boolean`: implemented as a 2-way constrained choice (`"true"` /
     `"false"`).

Because invalid tokens are never selected, the resulting JSON is always
syntactically valid and schema-compliant by construction — validity does
not depend on the model "getting it right" on its own.

## Design Decisions

- **`llm.py`**: the only module that talks to `llm_sdk.Small_LLM_Model`
  directly (SDK Facade). It also owns the `id -> token string` map, built
  once from the vocab file instead of once per request.
- **`decoder.py`**: pure constrained-decoding algorithms, independent of
  what the function/parameter schema actually is.
- **`model.py`**: orchestrates one full prediction — builds the natural
  language context sent to the model, calls the right decoder per
  parameter type, and assembles the typed result.
- **`models.py`**: pydantic `BaseModel` classes act as validated data
  contracts at every boundary (input files, output file).
- **`io_utils.py`**: isolates all file I/O and JSON error handling so the
  rest of the pipeline can assume clean, validated data.
- Each prompt is processed inside its own `try/except` in `main.py`, so a
  single failing prompt is skipped and logged instead of crashing the
  whole run.

## Performance Analysis

- **Correctness**: function selection and numeric extraction were tested
  against the provided sample prompts and were consistently accurate when
  a matching function was defined in `functions_definition.json`.
- **Speed**: each generation step calls `get_logits_from_input_ids`, which
  runs a full forward pass on the model; the current implementation
  additionally scans the full logits vector in plain Python at each step,
  which is not optimized for speed. This is acceptable for the required
  scale but is the main target for future optimization (bonus part).
- **Reliability**: output is always syntactically valid JSON, since
  invalid tokens are never selected — validity is structural, not
  probabilistic.

## Challenges Faced

- **Environment setup**: `llm_sdk` depends on `torch`, whose CUDA
  dependencies are large; this required configuring `uv` to fetch a
  CPU-only build and relocating caches/venv to a larger partition.
- **Runaway numeric generation**: an early version of the number decoder
  kept appending digits until an arbitrary cap was hit, because it only
  looked at the best score *among numeric tokens*, without checking
  whether the model actually still wanted to write a digit. This was
  fixed by comparing against the model's genuine, unconstrained top
  choice at each step.
- **Unbounded string generation**: string values initially kept
  rambling well past the intended answer because the only stop
  condition was a literal quote character, which never naturally
  appears in a plain-text (non-chat-templated) completion. Adding a
  newline-token stop condition fixed this.

## Testing Strategy

Manual testing was performed by running the pipeline against the sample
`function_calling_tests.json` and `functions_definition.json`, and
inspecting `data/output/function_calling_results.json` for schema
compliance and semantic correctness. Recommended next step: unit tests
(`pytest`) with `LLMWrapper` mocked so the decoding logic (choice /
number / string generators) can be tested deterministically without
loading the real model.

`flake8` and `mypy` (with the flags required by the subject) pass with
zero errors on the `src/` code. Four `mypy` errors remain inside the
vendored `llm_sdk` package itself (`Returning Any from function declared
to return "str"`); this file is provided as-is and is not meant to be
modified.

## Example Usage

```bash
uv run python -m src
```

Output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

## Resources

- Qwen model card: https://huggingface.co/Qwen/Qwen3-0.6B

**AI usage disclosure**: AI (Claude) was used throughout this project for:
explaining the project requirements and the constrained decoding concept;


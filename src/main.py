"""Entry point: load inputs, run the pipeline on every prompt, save results."""

import argparse
import time

from .io_utils import load_function_definitions, load_prompts, save_resault
from .llm import LLMWrapper
from .model import predict_function_call
from .models import FunctionCallResult


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the function-calling demo.

    Returns:
        argparse.Namespace: Parsed values for the function definition file,
            prompt input file, and output file.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )

    return parser.parse_args()


def main() -> None:
    """Execute the end-to-end prompt processing pipeline.

    Loads the function catalog and prompts, predicts the matching
    function calls, and writes the results to disk. Any single prompt
    error is skipped without
    aborting the rest of the run.
    """
    args = parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error loading input files: {error}")
        return

    print(
        "Loading model, this can take a while the first time..."
    )
    llm = LLMWrapper()
    start_time = time.perf_counter()
    results: list[FunctionCallResult] = []
    for item in prompts:
        try:
            result = predict_function_call(llm, item.prompt, functions)
            results.append(result)
            print(
                f"OK: {item.prompt!r} -> {result.name}({result.parameters})"
            )
        except ValueError as error:
            print(f"Skipping prompt {item.prompt!r}: {error}")

    try:
        save_resault(args.output, results)
    except OSError as error:
        print(f"Error saving results: {error}")
        return

    elapsed = time.perf_counter() - start_time
    print(f"Done: wrote {len(results)} results to {args.output}")
    print(f"Processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    """Run the function-calling demo from the command line."""
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as error:
        print(f"Unexpected error: {error}")

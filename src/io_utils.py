import json
from pathlib import Path

from pydantic import ValidationError

from .models import FunctionCallResult, FunctionDefinition, PromptItem


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Load and validate the available function definitions from JSON.

    Args:
        path: Path to a JSON file containing function schema definitions.

    Returns:
        list[FunctionDefinition]: Parsed function definitions.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or schema validation fails.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
        func = []
        for i in data:
            func.append(FunctionDefinition.model_validate(i))
        return func
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON: {path}")

    except ValidationError as e:
        raise ValueError(f"Invalid function definition: {e}")


def load_prompts(path: str) -> list[PromptItem]:
    """Load prompt records from a JSON file and validate each one.

    Args:
        path: Path to the JSON prompt dataset.

    Returns:
        list[PromptItem]: Prompt objects ready for inference.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or prompt validation fails.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)

        return [
            PromptItem.model_validate(item)
            for item in data
        ]

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON: {path}")

    except ValidationError as e:
        raise ValueError(f"Invalid prompt: {e}")


def save_resault(path: str, results: list[FunctionCallResult]) -> None:
    """Write function-call results to a JSON file.

    Args:
        path: Destination path for the output JSON.
        results: Predicted function-call results to serialize.

    Raises:
        OSError: If the output file cannot be created or written.
    """
    data = [result.model_dump() for result in results]
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(data, file, indent=2)
    except OSError as error:
        raise OSError(f"Could not write file: {path}") from error

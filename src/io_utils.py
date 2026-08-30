import json
from pathlib import Path

from pydantic import ValidationError

from .models import FunctionCallResult, FunctionDefinition, PromptItem


def load_function_definitions(path: str) -> list[FunctionDefinition]:
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
    data = [result.model_dump() for result in results]
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(data, file, indent=2)
    except OSError as error:
        raise OSError(f"Could not write file: {path}") from error

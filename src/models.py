from typing import Any

from pydantic import BaseModel, ConfigDict


class FunctionDefinition(BaseModel):
    """Description of a callable function exposed to the model.

    Attributes:
        name: Function name as it should be invoked.
        description: Human-readable explanation of the function's purpose.
        parameters: Mapping of argument names to their JSON-like schema.
        returns: Schema describing the value returned by the function.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


class PromptItem(BaseModel):
    """Single user prompt used as input for function-call prediction.

    Attributes:
        prompt: Raw prompt text supplied to the model.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):
    """Predicted function call produced for one prompt.

    Attributes:
        prompt: Original user request used to generate the prediction.
        name: Chosen function name.
        parameters: Argument values passed to that function.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]

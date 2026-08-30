from typing import Any
from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


class PromptItem(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]

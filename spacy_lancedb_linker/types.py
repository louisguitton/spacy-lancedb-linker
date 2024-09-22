from typing import Optional

from pydantic import BaseModel


class Entity(BaseModel):
    entity_id: str
    name: str
    description: str
    label: Optional[str] = None


class Alias(BaseModel):
    alias: str
    entities: list[str]
    probabilities: list[float]

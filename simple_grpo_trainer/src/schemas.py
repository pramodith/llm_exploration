from enum import Enum

class ModelType(str, Enum):
    """Enum for model types."""
    Active = "active"
    Old = "old"
    Reference = "reference"
from dataclasses import dataclass
from re import S
from typing import Dict, List, Optional, Set


@dataclass
class PromptInfo:
    objects: Set[str]  # Set of object names
    objectWithCounts: Dict[str, int]
    aspectRatio: Optional[float] = None  # Aspect ratio of the layout

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            objects=set(data["objects"]),
            objectWithCounts=data["objectWithCounts"],
            aspectRatio=data["aspectRatio"],
        )

    def to_json(self) -> Dict:
        return {
            "objects": list(self.objects),
            "objectWithCounts": self.objectWithCounts,
            "aspectRatio": self.aspectRatio,
        }


class BasePromptHandler:
    def __init__(self):
        pass

    def get_info(self, prompts: List[str]) -> List[PromptInfo]:
        raise NotImplementedError

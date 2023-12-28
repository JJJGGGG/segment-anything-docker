from pydantic import BaseModel
from typing import List, Tuple, Optional

class SegmentBody(BaseModel):
    image: str
    box: Optional[Tuple[int, int, int, int]] = None
    input_points: Optional[List[Tuple[int, int]]] = None
    input_labels: Optional[List[Tuple[int, int]]] = None
    multimask_output: Optional[bool] = None

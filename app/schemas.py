from pydantic import BaseModel
from typing import List


class Detection(BaseModel):
    cls: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    detections: List[Detection]
    compliant: bool
    rider_id: str | None = None
    order_id: str | None = None

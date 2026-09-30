"""
YOLOv8 inference wrapper.

Loads a fine-tuned weights file if present (app/weights/best.pt), otherwise
falls back to a stock pretrained YOLOv8n model so the service still runs
out of the box for demos/testing.
"""
import os
from ultralytics import YOLO

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "best.pt")

# Classes that count as a compliance failure when detected.
FAIL_CLASSES = {"no_helmet", "package_damaged"}

# Minimum confidence to trust a detection.
CONF_THRESHOLD = 0.5


class DeliveryDetector:
    def __init__(self, weights_path: str = WEIGHTS_PATH):
        if os.path.exists(weights_path):
            self.model = YOLO(weights_path)
        else:
            # Fallback so the API is runnable before you've trained a
            # custom model. Swap in your fine-tuned weights for real use.
            self.model = YOLO("yolov8n.pt")

    def predict(self, image_path: str):
        results = self.model.predict(
            source=image_path, conf=CONF_THRESHOLD, verbose=False
        )
        detections = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(
                    {
                        "cls": names[cls_id],
                        "confidence": float(box.conf[0]),
                        "box": [float(v) for v in box.xyxy[0].tolist()],
                    }
                )

        compliant = not any(d["cls"] in FAIL_CLASSES for d in detections)
        return detections, compliant


detector = DeliveryDetector()

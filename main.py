# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import modelclass



class YOLO(modelclass.Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        # Continue with default YOLO initialization
        super().__init__(model=model, task=task, verbose=verbose)


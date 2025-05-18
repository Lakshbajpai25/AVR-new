import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64

class ObjectDetector:
    def __init__(self):
        # Load YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect_objects(self, frame_data):
        try:
            # Convert base64 to image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform detection
            results = self.model(image)
            
            # Process results
            detections = []
            for *box, conf, cls in results.xyxy[0]:
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box.tolist()
                    label = self.class_names[int(cls)]
                    detections.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                        'label': label,
                        'confidence': float(conf)
                    })
            
            return {
                'success': True,
                'objects': detections
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            } 
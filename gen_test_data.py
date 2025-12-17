from ultralytics import YOLO
import json
import os
import numpy as np

model = YOLO('best.pt')
test_images_dir = 'test_data/images'
expected_results = {}

num_runs = 3

for img_name in sorted(os.listdir(test_images_dir)):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_images_dir, img_name)
        
        detections_list = []
        confidences_list = []
        
        for run in range(num_runs):
            results = model.predict(
                source=img_path, 
                conf=0.25, 
                iou=0.45,
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes
            num_detections = len(boxes)
            detections_list.append(num_detections)
            
            if num_detections > 0:
                confidences = boxes.conf.cpu().numpy()
                confidences_list.append(float(confidences.max()))
        
        unique, counts = np.unique(detections_list, return_counts=True)
        most_common_detections = int(unique[np.argmax(counts)])
        
        avg_confidence = np.mean(confidences_list) if confidences_list else 0.0
        min_confidence = max(0.4, avg_confidence * 0.9) if confidences_list else 0.5
        fire_detected = most_common_detections > 0
        
        expected_results[img_name] = {
            "fire_detected": fire_detected,
            "num_detections": most_common_detections,
            "min_confidence": round(min_confidence, 2)
        }

output_path = 'test_data/expected_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(expected_results, f, indent=2, ensure_ascii=False)
import os
import sys
import json
import numpy as np
from pathlib import Path

try:
    if not os.path.isfile("best.pt"):
        print("Model file 'best.pt' not found in root directory")
        print("Make sure the model is downloaded from the repository")
        raise Exception('Error: best.pt not found')

    if not os.path.isdir("test_data"):
        print("You need to create 'test_data' folder with test images")
        raise Exception('Error: test_data folder not found')
    
    if not os.path.isdir("test_data/images"):
        print("You need to create 'test_data/images' folder with test images")
        raise Exception('Error: test_data/images folder not found')
    
    test_images = [f for f in os.listdir("test_data/images") if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(test_images) == 0:
        print("No test images found in 'test_data/images' folder")
        raise Exception('Error: no test images found')

    if not os.path.isfile("test_data/expected_results.json"):
        print("You need to create 'test_data/expected_results.json' with expected detection results")
        print("Format: {\"image_name.jpg\": {\"fire_detected\": true, \"num_detections\": 1, \"min_confidence\": 0.5}}")
        raise Exception('Error: expected_results.json not found')

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics library is not installed")
        print("Install it: pip install ultralytics")
        raise Exception('Error: ultralytics not installed')
    
    print("Attempting to start fire detection model testing")

    print("\n[1/3] Loading YOLOv8 model...")
    model = YOLO('best.pt')
    print("Model loaded successfully")

    print("\n[2/3] Loading expected results...")
    with open("test_data/expected_results.json", 'r') as f:
        expected_results = json.load(f)
    print(f"Loaded {len(expected_results)} expected results")

    print("\n[3/3] Running detection on test images...")
    print(f"Found {len(test_images)} test images")
    
    results_summary = []
    all_tests_passed = True
    
    for img_name in test_images:
        img_path = os.path.join("test_data/images", img_name)
        print(f"\n  Processing: {img_name}")

        results = model.predict(
            source=img_path,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes
        
        num_detections = len(boxes)
        fire_detected = num_detections > 0
        confidences = boxes.conf.cpu().numpy() if fire_detected else np.array([])
        max_confidence = float(confidences.max()) if fire_detected else 0.0
        
        print(f"    - Detected objects: {num_detections}")
        if fire_detected:
            print(f"    - Maximum confidence: {max_confidence:.3f}")

        if img_name in expected_results:
            expected = expected_results[img_name]
            test_passed = True

            if expected.get("fire_detected") != fire_detected:
                print(f"ERROR: Expected fire_detected={expected.get('fire_detected')}, got {fire_detected}")
                test_passed = False
                all_tests_passed = False

            if "num_detections" in expected:
                expected_num = expected["num_detections"]
                tolerance = max(2, int(expected_num * 0.3))
                if abs(num_detections - expected_num) > tolerance:
                    print(f"ERROR: Expected {expected_num}±{tolerance} detections, got {num_detections}")
                    test_passed = False
                    all_tests_passed = False
                elif abs(num_detections - expected_num) > 0:
                    print(f"Warning: Expected {expected_num} detections, got {num_detections} (within tolerance ±{tolerance})")

            if fire_detected and "min_confidence" in expected:
                min_conf = expected["min_confidence"]
                if max_confidence < min_conf:
                    print(f"ERROR: Confidence {max_confidence:.3f} below threshold {min_conf}")
                    test_passed = False
                    all_tests_passed = False
            
            if test_passed:
                print(f"Test passed")
            
            results_summary.append({
                "image": img_name,
                "passed": test_passed,
                "detections": num_detections,
                "max_confidence": max_confidence
            })
        else:
            print(f"! Warning: no expected data for {img_name}")
    
    passed_tests = sum(1 for r in results_summary if r["passed"])
    total_tests = len(results_summary)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if all_tests_passed and total_tests > 0:
        print("\nTEST COMPLETE - ALL TESTS PASSED")
    else:
        print("\nTEST FAILED - SOME TESTS DID NOT PASS")
        sys.exit(1)

except Exception as e:
    print("TEST FAILS!!!!!!!")
    sys.exit(1)
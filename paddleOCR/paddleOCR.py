import json
import os
import cv2
from tqdm import tqdm
from paddleocr import PaddleOCR

# Paths
dataPath = r"C:\Users\cyril\Desktop\New folder (2)\drive-download-20250215T035812Z-001"
outputPath = r"C:\Users\cyril\Desktop\New folder (2)\Paddleoutput"
cocoFilePath = r"C:\Users\cyril\Desktop\New folder (2)\PaddleCOCO"

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

for index, subdir in enumerate(tqdm(os.listdir(dataPath), desc="Processing files in directory ")):
    coco_file = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "word"}]
    }
    
    imageID = 0
    annotationsID = 1
    
    for file in tqdm(os.listdir(os.path.join(dataPath, subdir)), desc="Processing files in subdirectory " + subdir):
        image_path = os.path.join(dataPath, subdir, file)
        image = cv2.imread(image_path)

        # Convert image to grayscale for better OCR performance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # OCR Processing
        results = ocr.ocr(gray, cls=True)
        
        imageID += 1
        coco_file["images"].append({
            "id": imageID,
            "file_name": file,
            "height": image.shape[0],
            "width": image.shape[1]
        })
        
        for line in results:
            for word_info in line:
                bbox, (text, confidence) = word_info
                
                x1, y1 = map(int, bbox[0])  # Top-left corner
                x2, y2 = map(int, bbox[2])  # Bottom-right corner
                w, h = x2 - x1, y2 - y1
                
                # Draw bounding box for each word
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Store annotation
                coco_file["annotations"].append({
                    "id": annotationsID,
                    "image_id": imageID,
                    "category_id": 1,
                    "attributes": {"text": text},
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotationsID += 1
        
        # Save processed image
        output_subdir = os.path.join(outputPath, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        cv2.imwrite(os.path.join(output_subdir, "output_" + file), image)

    # Save COCO JSON
    coco_subdir = os.path.join(cocoFilePath, subdir)
    os.makedirs(coco_subdir, exist_ok=True)
    with open(os.path.join(coco_subdir, "coco" + str(index) + ".json"), "w") as f:
        json.dump(coco_file, f, indent=4)

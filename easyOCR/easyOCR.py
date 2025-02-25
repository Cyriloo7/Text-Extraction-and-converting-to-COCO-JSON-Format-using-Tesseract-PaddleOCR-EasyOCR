# EasyOCR is used to extract text from images and draw bounding boxes around the text.

import json
import os
import cv2
import easyocr
from tqdm import tqdm

dataPath = r"C:\Users\cyril\Desktop\New folder (3)\New folder (2)\drive-download-20250215T035812Z-001"
outputPath = r"C:\Users\cyril\Desktop\New folder (3)\New folder (2)\output"
cocoFilePath = r"C:\Users\cyril\Desktop\New folder (3)\New folder (2)\COCO"

reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

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
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        results = reader.readtext(
            gray,
            detail=1
        )

        imageID += 1
        coco_file["images"].append({
            "id": imageID,
            "file_name": file,
            "height": image.shape[0],
            "width": image.shape[1]
        })

        for result in results:
            if len(result) < 3:
                continue  # Skip invalid results
            bbox, text, conf = result
            
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue  # Ensure bounding box has exactly 4 points

            x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
            x_max, y_max = int(bbox[2][0]), int(bbox[2][1])
            w, h = max(1, x_max - x_min), max(1, y_max - y_min)  
           

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            annotation = {
                "id": annotationsID,
                "image_id": imageID,
                "category_id": 1,
                "attributes": {"text": text},
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            }
            coco_file["annotations"].append(annotation)
            annotationsID += 1

        output_subdir = os.path.join(outputPath, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        cv2.imwrite(os.path.join(output_subdir, "output_" + file), image)

    coco_subdir = os.path.join(cocoFilePath, subdir)
    os.makedirs(coco_subdir, exist_ok=True)
    with open(os.path.join(coco_subdir, "coco" + str(index) + ".json"), "w") as f:
        json.dump(coco_file, f, indent=4)

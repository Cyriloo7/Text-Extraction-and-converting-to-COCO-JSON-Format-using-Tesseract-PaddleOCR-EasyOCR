import json
import os
import pytesseract
import cv2
from tqdm import tqdm

dataPath = r"C:\Users\cyril\Desktop\New folder\drive-download-20250215T035812Z-001"
outputPath = r"C:\Users\cyril\Desktop\New folder\output"
cocoFilePath = r"C:\Users\cyril\Desktop\New folder\COCO"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


for index, subdir in enumerate(tqdm(os.listdir(dataPath), desc="Processing files in directory ")):

    coco_file = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_file["categories"].append({
        "id": 1,
        "name": "word"
    })

    imageID = 0
    categoryList = []
    annotationsID = 1
    for file in tqdm(os.listdir(os.path.join(dataPath, subdir)), desc="Processing files in subdirectory " + subdir):
        image_path = os.path.join(dataPath, subdir, file)
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's Thresholding
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # OCR Config
        custom_config = r'--oem 3 --psm 1'  # OCR Engine Mode 3, Page Segmentation Mode 1
        detection_results = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

        imageID += 1
        coco_file["images"].append({
            "id": imageID,
            "file_name": file,
            "height": image.shape[0],
            "width": image.shape[1]
        })
        for i in range(len(detection_results["text"])):
            annotations = {
                "id": -1,
                "image_id": -1,
                "category_id": -1,
                "attributes":{
                    "text": ""
                },
                "bbox": [],
                "iscrowd": 0,
                "segmentation": [],
                "area": -1,
            }
            if detection_results["text"][i].strip():  # Ignore empty detections
                x, y, w, h = (detection_results["left"][i], detection_results["top"][i], 
                            detection_results["width"][i], detection_results["height"][i])
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(image, detection_results["text"][i], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                annotations["id"] = annotationsID
                annotations["image_id"] = imageID
                annotations["category_id"] = 1
                annotations["attributes"]["text"] = detection_results["text"][i].strip()
                annotations["bbox"] = [x, y, w, h]
                annotations["area"] = w * h

                annotationsID += 1

                coco_file["annotations"].append(annotations)
        if not os.path.exists(os.path.join(outputPath, subdir)):
            os.makedirs(os.path.join(outputPath, subdir))
        
        cv2.imwrite(os.path.join(outputPath, subdir, "output_" + file), image)


    if not os.path.exists(os.path.join(cocoFilePath, subdir)):
        os.makedirs(os.path.join(cocoFilePath, subdir))

    with open(os.path.join(cocoFilePath, subdir, "coco" + str(index) + ".json"), "w") as f:
        json.dump(coco_file, f, indent=4)


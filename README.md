# Text Extraction and Conversion to COCO JSON Format

This repository provides scripts to extract text from images using Optical Character Recognition (OCR) tools—**Tesseract**, **PaddleOCR**, and **EasyOCR**—and convert the extracted text into the COCO JSON format.

## Features

- **Text Extraction**: Utilize Tesseract, PaddleOCR, or EasyOCR to extract text from images.
- **COCO JSON Conversion**: Convert extracted text into the COCO JSON annotation format for seamless integration with machine learning workflows.

## Prerequisites

- **Python 3.x**

- Install required packages:
  ```bash
  pip install pytesseract paddleocr easyocr opencv-python-headless

# Usage
## Clone the Repository:
```bash
git clone https://github.com/Cyriloo7/Text-Extraction-and-converting-to-COCO-JSON-Format-using-Tesseract-PaddleOCR-EasyOCR.git
cd Text-Extraction-and-converting-to-COCO-JSON-Format-using-Tesseract-PaddleOCR-EasyOCR
```

## Extract Text from Images:
### Using Tesseract:
python
Copy
Edit
import pytesseract
from PIL import Image

image = Image.open('path_to_image')
text = pytesseract.image_to_string(image)
### Using PaddleOCR:
python
Copy
Edit
from paddleocr import PaddleOCR

ocr = PaddleOCR()
result = ocr.ocr('path_to_image')
### Using EasyOCR:
python
Copy
Edit
import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext('path_to_image')
Convert to COCO JSON Format:

After extracting text, use the provided script to convert the results into COCO JSON format:
python
Copy
Edit
from utils import convert_to_coco

annotations = extract_text_annotations('path_to_images', ocr_tool='tesseract')
coco_format = convert_to_coco(annotations)
References
Tesseract OCR: GitHub Repository
PaddleOCR: GitHub Repository
EasyOCR: GitHub Repository
COCO JSON Format: COCO Dataset

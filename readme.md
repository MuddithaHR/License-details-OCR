# Driving License Detail Extractor #
A program to OCR an image of the driving license and extract vehicle categories with relevant issued and expiry dates.

- by Mudditha Ranathunga

---

## 🚀 Features

- 🔍 Automatically detects information table of the license in images using finetuned YOLOv5
- 🧠 Performs text extraction (OCR) from detected table region using PaddleOCR
- 🖼️ Supports input from local image files
- 📝 Outputs structured feedback or data from processed images
- 📦 Modular pipeline with separate model loading and processing


---


## 🚀 Program Setup 

Using python 3.8 version is recommended as newer versions may not be compatible with paddleocr library (I used 3.8.10)

### - Install requirements.txt ###
```bash
pip install -r requirements.txt

```

### - Run program
Now you can give image path to the 'img_file_path' in main.py and run it.

---


## 📁 Project Structure

```bash
project-name/
├── main.py               
├── pipeline.py           # contains processing pipeline
│
├── yolo_detection/       # contais .py files required to load YOLO and detect information table in lincense
│   └── __init__.py
│   └── load_model.py
│   └── detect_info_table.py
│   └── utils.py
│
├── ocr/                  # contais .py files required to load OCR model
│   └── __init__.py
│   └── load_ocr_model.py
│
├── postprocessing/       # contais .py files required for process OCR output (filter dates & categories, find image orientation, identify pairs)
│   └── __init__.py
│   └── filter_ocr.py
│   └── orientation.py
│   └── row_identification.py
│   └── utils.py
│
├── utils/                # contais .py files required for additional support functions
│   └── __init__.py
│   └── bounding_box_utils.py
│   └── config_loader.py
│   └── save_csv.py
│
├── outputs/              # contais .csv outputs by the program
│   └── Sample Data       # contains generated .csv files for given sample 99 images and their summary
│       └── results_summary    
├── configs/              # contais configurations
│   └── config.yaml
├── models/               # contais finetuned models
│   └── finetuned_yolo
│       └── best.pt       #finetuned yolo weights
│       └── Finetuning script and data      #Finetuning notebook and dataset
├── requirements.txt      # pip packages
└── README.md

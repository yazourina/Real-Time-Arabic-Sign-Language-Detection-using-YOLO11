# 🖼️ Digital Image Processing Final Project  
## Real-Time Arabic Sign Language (ArASL) Detection using YOLO11  

---

### 📖 Project Overview
This project implements **real-time object detection** for the **Arabic Sign Language (ArASL) dataset** using **YOLO11** (Ultralytics).  
It is developed as part of the **Digital Image Processing final project**.  

- Dataset: https://kaggle.com/datasets/sabribelmadoui/arabic-sign-language-unaugmented-dataset 
- Classes:**28 characters** (Arabic alphabet signs)  
- Framework: Ultralytics YOLO11, RoboFlow, OpevCV
- Goal: Train, validate, and deploy a YOLO11 model for real-time detection on webcam feed.

## 🚀 Features
- Detects **28 Arabic Sign Language (ArASL) letters** in real-time
- Based on **YOLO11** object detection models (fast & accurate)
- Supports:
  - Training on custom datasets
  - Validation (mAP, precision, recall)
  - Inference on images, videos, and webcam
- Export to multiple formats: **TorchScript, ONNX, TensorRT, CoreML, TFLite**

## 🧠 Model
- Architecture: **YOLO11n** (pretrained on COCO, fine-tuned on ArASL dataset)
- Dataset: **ArASL** (28 classes, 4k+ images)
- Training setup:
  - Epochs: 30 (adjustable)
  - Image size: 640×640
  - Batch size: 32 (can be tuned)

## 🛠 Installation

### Requirements
- Python 3.9+ (tested with Python 3.11)
- [Ultralytics](https://pypi.org/project/ultralytics/)
- OpenCV
- PyTorch (with CUDA if GPU available)

### Install Dependencies
```bash
pip install ultralytics opencv-python torch


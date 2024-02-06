# YOLO8 Segmentation Deployment Using TensorRT and ONNX

This repository offers a production-ready deployment solution for YOLO8 Segmentation using TensorRT and ONNX. It aims to provide a comprehensive guide and toolkit for deploying the state-of-the-art (SOTA) YOLO8-seg model from Ultralytics, supporting both CPU and GPU environments. Our focus is to ensure seamless integration of AI models, specifically the cutting-edge YOLOv8, into [Unitlab Annotate](http://unitlab.ai/) for enhanced object detection, instance segmentation, and other AI-driven tasks.

---

[![Python Version](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![img](https://badgen.net/badge/icon/tensorrt?icon=azurepipelines&label)](https://developer.nvidia.com/tensorrt)


---

## About YOLOv8

YOLOv8 represents the latest advancement in the YOLO series, developed by Ultralytics. It is a culmination of ongoing research and development, pushing the boundaries of speed, accuracy, and efficiency in object detection and segmentation. YOLOv8 is designed to excel in a wide range of tasks including object detection, tracking, instance segmentation, image classification, and pose estimation. For more details on YOLOv8, visit the official Ultralytics GitHub page: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).


## About Unitlab Annotate
AI-powered platform to manage, label, and enhance your data. Help your machine learning teams to automate the data annotation workflow. Bring your own pre-trained models to Unitlab Annotate and annotate your data 15 times faster. [Get started right now.](http://unitlab.ai/)

## Features

- **Fast and Efficient**: Optimized production settings with TensorRT and ONNX, supporting both CPU and GPU options.
- **Production-Ready**: It requires just one line of command for production deployment.
- **Integration with Unitlab Annotate**: Demonstrates how to integrate your AI models with Unitlab Annotate for enhanced annotation capabilities. Follow up how to integrate it in https://docs.unitlab.ai/ai-models/model-integration

## Getting Started

To get started with deploying YOLO8 Segmentation, follow the steps outlined below:

### Prerequisites

Ensure you have the following installed:

- Docker (for containerized deployment)
- Python 3.8 or higher
- Docker Compose (v2.20.2)
- TensorRT (for GPU acceleration)
- ONNX Runtime (for CPU deployment)

### Installation

1. Clone the Repository
   
Start by cloning this repository to your local system:

```bash
git clone https://github.com/teamunitlab/yolo8-segmentation-deploy.git
cd yolo8-segmentation-deploy
```

2.  Build & Run for CPU
   
Use Docker Compose to deploy for CPU environments:

```bash
docker-compose -f docker-compose-cpu.yml up -d --build
```

3.  Build & Run for GPU
   
For GPU acceleration, execute the following:

```bash
docker-compose -f docker-compose-gpu.yml up -d --build
```

4.  Local Testing
   
Test the deployment locally to ensure everything is working correctly:



```bash
# Using cURL
curl -X POST http://localhost:8080/api/yolo8/coco-segmentation \
     -H "Content-Type: application/json" \
     -d '{"src":"https://t3.ftcdn.net/jpg/02/43/12/34/360_F_243123463_zTooub557xEWABDLk0jJklDyLSGl2jrr.jpg"}'
```

```bash
# Using a Python Script
python service_test.py
```

### Licence
MIT

### References
1. https://github.com/ultralytics/ultralytics
2. https://docs.unitlab.ai/ai-models/model-integration
3. https://github.com/triple-Mu/YOLOv8-TensorRT

# Terrain Recognition: Object Detection Pipeline

## Overview

Terrarian Recognition is a modular, reproducible object detection pipeline trained in the xView dataset, built with PyTorch. It supports multiple model architectures (Faster R-CNN, ELU+Focal variants), robust data preprocessing, training, evaluation, and inference. The project is cloud-ready and follows best practices for research and production.

---
## Contributors
- Developed by [a1regg](https://github.com/a1regg)
- Trained on AWS by [Slowlybomb](https://github.com/Slowlybomb)

---
## Features
- **Modular codebase**: Clean separation of data, models, utils, and config
- **Multiple models**: Standard Faster R-CNN and custom ELU+Focal variant
- **Configurable**: All settings via `.env` and `src/config.py`
- **Reproducible**: Deterministic seeding, Docker support
- **Cloud-ready**: Train/evaluate on AWS or locally (CPU/GPU)

---

## Project Structure

```
├── Dockerfile
├── readme.md
├── .env
├── requirements.txt
├── data/
│   ├── xView_train.geojson
│   └── dataset/
│       ├── train_images/
│       └── tiled_images/
├── models/
│   └── fasterrcnn # and others also here
│       ├── model.pth
│       └── model_metrics.pkl
├── src/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   └── models/
│       ├── __init__.py
│       ├── faster_rcnn
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── enhanced.py
│       └── yolo
            ├── __init__.py
            └── yolo.py 
```

---


## Start
### It is strongly recommended to train on a cloud service (e.g. AWS), as the model is too large. However, you can use your own computer; although I won't explain how, just run `train.py`.
Build and run the container:
```sh
docker build -t terrain_recognition .
docker run --gpus all --shm-size=<int>g -it terrarian_recognition 
```

---

## Configuration
- All settings (batch size, learning rate, model, etc.) are in `.env` and loaded by `src/config.py`.


---

## Model Architectures
- **Base**: Faster R-CNN Standard PyTorch implementation
- **Enhanced** **Faster R-CNN + ELU + Focal Loss**: Custom variant for improved performance on imbalanced data
- **Ultimate**: Faster R-CNN + ELU + Class-Balanced FL + Attention mechanism + enhanced FPN with cross-scale feature fusion
- **Yolo**: base YOLO model and Enhanced: customized for better satellite imagery

To train Faster-RCNN models, use `train.py` with `--model` tag as shown below
To train YOLO models, use `train_yolo.py`.

#### Example usage

```
# Faster R-CNN
python src/train.py                  # Faster-RCNN, which is a base model, is the default, but you can run --model base 
python src/train.py --model enhanced
python src/train.py --model ultimate 
                                 

# Yolo
python src/train_yolo.py --model-size {n, s, m, l, x} --enhanced 
# for more optional tags, write 
python src/train_yolo -h # or --help
```
---

### Evaluate
 To evaluate how model or models performed, run evaluate script:
```
python src/evaluate.py --model <model> --limit <int> # for one model, the limit specifies the number of images you would like to be processed. Omit this if you trained on the whole dataset.

# or for multiple models
python src/evaluate.py --models <model1> <model2>
```
### Inference
 To run **Inference** run `inference.py`:
- `--input` (required): Path to an image or directory of images.
- `--model` (optional): Model name. Defaults to the value in `.env` or `fasterrcnn`(base model).
- `--output` (optional): Output directory for annotated images. Defaults to `data/output`.
- `--score-threshold` (optional): Minimum score for displaying detections (default: 0.3).
Example usage:
```
python src/inference.py --input data/dataset/train_images/ --model fasterrcnn --output my_results/
```
## Acknowledgements
- [xView Dataset](https://xviewdataset.org/)
- PyTorch, torchvision, and the open-source community

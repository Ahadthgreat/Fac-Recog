# YOLOv8 Face Recognition

This repository contains a face recognition model based on YOLOv8, trained on a dataset of over 13,000+ training images and 3,000+ validation images. The model aims to accurately detect faces in images and videos.

## Setup

### Prerequisites

- Python 3.9

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Adiiii02/Face-Recognition-YOLO.git
   cd Face-Recognition-YOLO
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Model

Download the pre-trained model `best.pt` and place it in the root directory of the repository.

## Usage

### Predicting on a Video

You can use the model to predict faces in a video file as follows:

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

results = model.predict(source="demo.mp4", show=True)  # Use Example Video/Webcam in Source

print(results)
```

### Improving Accuracy

The model's accuracy can be improved by:
- Increasing the number of training epochs.
- Using data augmentation techniques.
- Hyperparameter tuning.
- Increasing the dataset size.
- Applying regularization techniques.

## Training

The model was trained on Kaggle using the following parameters:
- **Number of Epochs**: 7
- **Training Images**: 13,000+
- **Validation Images**: 3,000+

To train the model, refer to the IPYNB Notebook that sets up the YOLOv8 model and handles the training loop.

## Results

The following metrics were achieved:
- **Training mAP**: 0.872
- **Validation mAP**: 0.866
- **Training Precision/Recall**: 0.893 / 0.797
- **Validation Precision/Recall**: 0.884 / 0.79

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- YOLOv8 by Ultralytics
- Kaggle for providing the training and validation datasets

---

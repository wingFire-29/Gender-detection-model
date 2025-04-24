#Gender Detection

Welcome to the **Gender Detection Model** project by [@wingFire-29](https://github.com/wingFire-29)!  
This project uses deep learning and OpenCV to detect human faces in real-time and predict their gender — Male or Female — with high confidence.


## Features

- Real-time face detection
- Gender classification with probability score
- Support for both webcam and image inputs
- Testing framework to evaluate model accuracy

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pre-trained models (included in the repository)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/age-and-gender-detection.git
   cd age-and-gender-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Real-time Detection

To run gender detection on your webcam:

```
python detect.py
```

To run detection on a specific image:

```
python detect.py --image path/to/image.jpg
```

Press 'q' to exit the application.

### Testing Model Accuracy

To evaluate the model accuracy on a dataset:

```
python testing1.py --dataset path/to/dataset
```

The dataset should have the following structure:
```
dataset/
  ├── Male/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── Female/
      ├── image1.jpg
      ├── image2.jpg
      └── ...
```

## Pre-trained Models

The project uses pre-trained models for face detection and gender classification:

- Face Detection: OpenCV's DNN-based face detector
- Gender Classification: Caffe model for gender recognition

## Dataset

For testing purposes, the project includes two datasets:
- dataset100: A small dataset with 100 images
- dataset1000: A larger dataset with 1000 images

## Results

Testing results show an accuracy of XX% on gender classification.

## License

[Add your license information here] 

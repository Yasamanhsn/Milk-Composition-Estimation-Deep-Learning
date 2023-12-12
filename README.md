# Milk Composition Estimation using Deep Learning

This repository contains code for estimating key milk components (SNF, Fat, Protein) from spectroscopy data using advanced deep learning techniques.

## Overview

Accurate measurement of milk components is crucial for maintaining food quality and meeting nutritional standards. This project focuses on leveraging deep learning methodologies to precisely quantify essential milk constituents.

## Methodology

### Data Preprocessing
The spectral data is converted into time-frequency scalograms to generate image representations.

### Model Architecture
A hybrid MobileNetV2 model, enriched with attention mechanisms, is implemented and optimized using a custom loss function.

## Usage

### Prerequisites
- Python 3.x
- TensorFlow 2.10
- NumPy 1.23
- Matplotlib 3.82

### Installation
1. Clone this repository: `git clone https://github.com/Yasamanhsn/Milk-Composition-Estimation-Deep-Learning.git`
2. Install dependencies: `pip install -r requirements.txt`

### Running the Code
1. Navigate to the project directory.
2. Run the main script: `python main.py`

## Results

The model achieved impressive accuracy with low estimation errors across components. Mean squared errors were measured at 0.2716, maximum error value at 1.3102, and R-squared above 0.831.

For more detailed information, refer to the paper ["Estimation of Milk Composition from Spectroscopy 
Data Using Advanced Deep Learning Model"].

## Contributors

- t Yasaman Hosseini - writter


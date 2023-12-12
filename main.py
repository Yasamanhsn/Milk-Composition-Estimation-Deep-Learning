"""
Spectrometry Analysis of Milk

This code is designed for spectrometry analysis of liquid milk using visible light and UV spectrometers. The process
involves the conversion of obtained spectra from a CSV file to the frequency domain, achieved through wavelet
transformations. The resulting time-frequency images are then saved on the computer for further analysis.

To address computational limitations, the code leverages MobileNet Version 2, a deep neural network architecture.
This version is enhanced with an attention layer, providing improved power and adaptability. The attention mechanism
 helps the network focus on critical features, making it particularly effective for the analysis of complex data.

The primary objective of this code is to detect and quantify key components in liquid milk, specifically fat content,
Solids-Not-Fat (SNF), and milk protein. The utilization of MobileNet Version 3 with an attention layer enhances
the accuracy of these detections, ensuring precise and reliable results.

This research code is the output of my dissertation for a Master's in Artificial Intelligence, focusing on the development
and application of advanced neural network models for spectroscopic analysis in the food industry.

Author:
Yasaman Hosseini

Email:
yasaman_hosseini@mail.um.ac.ir

Affiliations:
Department of Computer Engineering,
Faculty of Engineering Ferdowsi University of Mashhad,
9177948974 Mashhad, Iran.
"""


from TrainNetwork.LoadImages import load_images
from TrainNetwork.PreprocessData import preprocess_data
from TrainNetwork.BuildModel import build_model
from TrainNetwork.FitModel import fit_model
from TrainNetwork.EvaluateModel import evaluate_model
from TrainNetwork.LoadSignalAndConvertToImage import load_signal_and_convert_to_image
import time

dataset_folder_name = 'dataset'
csv_file_name = 'NewTrain21.csv'
target_file_name = 'target.csv'

# Convert the input signals into scalogram images
load_signal_and_convert_to_image(dataset_folder_name, csv_file_name, target_file_name)

# Load images and target data
images, targets = load_images(dataset_folder_name, target_file_name)

# Preprocess data
is_use_from_saved_data=False
X_train, X_test, y_train, y_test = preprocess_data(images, targets,is_use_from_saved_data)


# Build and train the model
# , 'simple_cnn'
models = ['hybrid_mobilenet_v2_attention',
         'mobilenet_v2', 'mobilenet_v3',
          'inception_resnet_v2', 'vgg16','vgg19','resnet50']
metric={}
# Build and train the model
for model_name in models:
    print(f'\n\nTraining {model_name}')
    if model_name =="simple_cnn":
        is_use_custom_loss = True
        batch_size = 8
    else:
        is_use_custom_loss = False
        batch_size = 16


    model = build_model(model_name, is_use_custom_loss)

    # Start timer
    start_time = time.time()

    model = fit_model(model, X_train, y_train,
                      epochs=3000, batch_size=batch_size, validation_split=0.2,
                      model_name=model_name)
    # End Timer
    end_time = time.time()
    tr_time = end_time-start_time

    # save metrics
    metric[model_name] = {'train_time': tr_time,
                          'num_params':model.count_params()}


    # evaluate_model(model, X_test, y_test, model_name,metric)

    params = model.summary()

print(metric)
print('All models trained and evaluated!')




# Medical Image Analysis and Classification

## AIMS & OBJECTIVES:

The aim of this project is to develop two Convolutional Network models, a breast cancer prediction model and a blood cell classifier that can identify 8 distinct types of blood cells, for medical diagnosis. I was required to use BloodMNIST and BreastMNIST datasets publicly available at https://medmnist.com, to train multiple Convolutional Neural Networks (CNNs), using a combination of self-designed CNN architectures and pretrained networks like VGG16 (used in blood cells classifier model). To address class imbalance issues, I proposed a solution in our report in the implementation of Breast Cancer Model using synthetic data

The end goal is to create a web interface that allows users to upload medical data, choose the among the two classifiers (breast cancer or blood cell classifier) and receive a prediction on whether they have breast cancer or identify the type of blood cell based on the model selected. To ensure the reliability and generalization of our models, I have extensively evaluated our model’s performance using classification metrics such as Precision, Recall, F1-Score and Weighted accuracies.

## RUNNING WEB APPLICATION

To run the web application, download all dependencies from requirements.txt and then run following command in the cli of project directory:

```bash
  streamlit run app.py
```

## Documentation

Project directory has 10 different files and folders. 

### 1. breastmnist

This folder contains,

1. The model weights with optimal performance, over a range of epochs, for each of our model created. These are named as "self_model.weights.best.hdf5","aug_model.weights.best.hdf5"and "model_resampled.weights.best.hdf5".

2. The notebook for analysis and creation of model using breastmnist dataset, saved as "breastmnist.ipynb".

3. Data Folder containing the breastmnist dataset in separates folders for train,test and validation splits respectively.

### 2. bloodmnist

This folder contains,

1. The model weights with best optimal performance over a certain range of epochs, for each of our model created. These are named as "self_model.weights.best.hdf5","aug_model.weights.best.hdf5"and "vgg_model.weights.best.hdf5".

2. The notebook for analysis and creation of model using breastmnist dataset, saved as "bloodmnist.ipynb".

3. Data Folder containing the bloodmnist dataset in separate folders for train,test and validation splits respectively.

### 3. Blood Cells Sample Images

This folder contains blood cells sample images for testing purposes.

### 4. Breast Cancer Sample Images

This folder contains breast cancer sample images for testing purposes.

### 5. Blood_cells_labelled_samples

This folder contains labelled blood cells types for testing purposes.

### 6. COURSE WORK-2-Group-14-Report.docx (Project Final Report) 

Project Final Report can be found in "COURSE WORK-2-Group-14-Report.docx".

### 7. model_blood.h5

This is the best model selected for the blood cell classifier.

### 8. model_breast.h5

This is the best model selected for breast cancer model.

### 9. requirements.txt

This file contains the dependencies needed to run our web application.

### 10. app.py

This file contains the code needed to run our best models as web application.










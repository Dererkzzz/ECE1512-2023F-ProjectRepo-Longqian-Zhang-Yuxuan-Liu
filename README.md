## Datasets

MNIST:  
Description: A digit classification dataset with 10 classes (0-9).  
Data Size: 60,000 training data + 10,000 test data.  

Minimalist HIStopathology (MHIST):  
Description: A binary classification task for colorectal polyp images.  
Data Size: 2,175 training data + 977 test data.  
 

## Prerequisites:
Python 3.X  
PyTorch  
Torchvision  
NumPy  
Pandas  
Matplotlib  
Seaborn  
SciPy  

# Project B - Data -- Longqian Zhang, Yuxuan Liu



 































# Project A - Knowledge Distillation for Model Compression -- Longqian Zhang, Yuxuan Liu


## Motivation

Deep neural networks (DNNs) have been widely deployed on the cloud for a wide spectrum of applications, from computer vision to natural language processing. However, deploying these large models on edge devices with limited resources can be challenging due to computational complexity and storage requirements. Knowledge distillation (KD) offers a solution to deploy lightweight models on such devices without sacrificing accuracy.

This project focuses on knowledge distillation as a model compression technique and is divided into two tasks. Task 1 uses conventional knowledge distillation on the MNIST digit classification dataset, while Task 2 applies transfer learning and knowledge distillation to train a lightweight model for a clinical histopathology dataset (MHIST).

 

## Task 1: Knowledge Distillation in MNIST Dataset  
Implementation of conventional knowledge distillation for MNIST.  
Compare the performance of the student with and without KD.  
![task2](/ProjectA/Picture2.jpg)  
Task 1. Result

## Task 2: Knowledge Distillation in MHIST Dataset  
Use pre-trained ResNet50V2 and MobileNetV2 networks for teacher and student models.  
Perform knowledge distillation.  
Evaluate model performance for an imbalanced dataset.  
![task2](/ProjectA/Picture1.png)    
Task 2. Result

## Reference  

More detail in ProjectA report.

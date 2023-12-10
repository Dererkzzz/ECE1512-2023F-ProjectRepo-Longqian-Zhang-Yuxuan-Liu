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
Pickle  
Logging  
Shutil  
Opacus  
# Project B - Data -- Longqian Zhang, Yuxuan Liu


## Motivation
 
In order to reduce the challenge of high computational sources for deep learning models trained on large-scale datasets, Wang et al. propose an alternative model-space based approach to reduce computational costs, namely dataset distillation (DD). The goal of this project to create a synthetic small S that has the most discriminative features of the original large-scale dataset T.


## Task 1: Knowledge Distillation in MNIST Dataset  
@inproceedings{
zhao2021DC,
title={Dataset Condensation with Gradient Matching},
author={Bo Zhao and Konda Reddy Mopuri and Hakan Bilen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=mSAKhLYLSsl}
}  
@inproceedings{chen2022privateset,
  title={Private Set Generation with Discriminative Information},
  author={Chen, Dingfan and Kerkouche, Raouf and Fritz, Mario},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2022}
}  

In this task, we used the dataset distillation with gradient matching to learn a synthetically small dataset for the MNIST and MHIST datasets, train networks from scratch on the condensed images, and then evaluate them on the real testing data. This is one of the fundamental frameworks for dealing with dataset distillation in computer vision classification tasks while decreasing the computational costs. 

## Task 2: Knowledge Distillation in MHIST Dataset  
@misc{guo2023lossless,
      title={Towards Lossless Dataset Distillation via Difficulty-Aligned Trajectory Matching}, 
      author={Ziyao Guo and Kai Wang and George Cazenavette and Hui Li and Kaipeng Zhang and Yang You},
      year={2023},
      eprint={2310.05773},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}   
@inproceedings{kimICML22,
title = {Dataset Condensation via Efficient Synthetic-Data Parameterization},
author = {Kim, Jang-Hyun and Kim, Jinuk and Oh, Seong Joon and Yun, Sangdoo and Song, Hwanjun and Jeong, Joonhyun and Ha, Jung-Woo and Song, Hyun Oh},
booktitle = {International Conference on Machine Learning (ICML)},
year = {2022}
}   

In this task, we would like to use the two state-of-the-art methods introduced in Section 1 of the Introduction and compare them to the "gradient matching" algorithm you used in Task 1 to further explore the effectiveness of dataset-refinement methods in visual classification tasks. The papers are "Towards Lossless Dataset Distillation via Difficulty-Aligned Trajectory Matching" and "Dataset Condensation via efficient synthetic-data parameterization."
 





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

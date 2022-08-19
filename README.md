# Unified-Ensemble-Averaging-Model

This project repository contains the overall implementation guideline, procedures, and partial dataset for the adaptive weighted ensemble averaging model. 

## Getting Started 

Before implementing the classification tasks, there are some pre-requisites to comply:

- Computational environment: Tensorflow 2.6.0, Python 3.7 at least, and GPU encouraged
- Data pre-processing completed:
  * All the images must be in the same size, in this project, we have an input size of 224*224*3
  * Data augmentation is encouraged if class imbalance issue happens (if necessary, please view [Data augmentation](https://github.com/Amyyy-z/Unified-Ensemble-Averaging-Model/blob/main/Data%20augmentation.py) for image rotating and flipping methods
- Samples of thyroid ultrasound images can be downloaded through [Dataset.zip](https://github.com/Amyyy-z/Unified-Ensemble-Averaging-Model/blob/main/Dataset.zip)

## Implementation Guide

* Prepare the image sets with the partial contributed datasets and the external data
* Import required libraries
* Import image sets with their labels: 0 indicates benign, and 1 indicates malignant
* Encode image labels with one-hot-encoding technique
* Training and Testing splits with training, validation, testing (8:1:1 ratio encouraged)
* Construct individual learner CNN models (VGG, ResNet, Inception, Xception, DenseNet)
* Feed the images and encoded labels into the model for training and fine-tuning, please refer to [Individual learner pre-training](https://github.com/Amyyy-z/Unified-Ensemble-Averaging-Model/blob/main/Individual%20learner%20pre-training.py) for pre-training individual networks implementation details
* Selected best-performing individual learner based on their performance (accuracy, auc, precision, recall, specificity, npv, f1, fpr)
* Assign weights to the best-performing model
* Construct the weighted ensemble averaging model, please refer to [Weighted ensemble averaging model](https://github.com/Amyyy-z/Unified-Ensemble-Averaging-Model/blob/main/Weighted%20ensemble%20averaging%20model.py) for the model implementation
* Evaluate the ensemble model with the external dataset


---------------------------

### Enjoy!

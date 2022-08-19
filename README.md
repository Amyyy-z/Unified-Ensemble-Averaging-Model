# Unified-Ensemble-Averaging-Model

This project repository contains the overall implementation guideline, procedures, and partial dataset for the adaptive weighted ensemble averaging model. 

## Getting Started 

Before implementing the classification tasks, there are some pre-requisites to comply:

- Computational environment: Tensorflow 2.6.0, Python 3.7 at least, and GPU encouraged
- Data pre-processing completed:
  * All the images must be in the same size, in this project, we have an input size of 224*224*3
- Samples of thyroid ultrasound images can be downloaded through [Dataset.zip](https://github.com/Amyyy-z/Unified-Ensemble-Averaging-Model/blob/main/Dataset.zip)

## Implementation Guide

* Prepare the image sets
* Import required libraries
* Import image sets with their labels: 0 indicates benign, and 1 indicates malignant
* Encode image labels
* Training and Testing splits with training, validation, testing (8:1:1 ratio encouraged)
* Construct individual learner CNN models
* Feed the images and encoded labels into the model for training and fine-tuning
* Selected best-performing individual learner based on their performance
* Assign weights to the best-performing model
* Construct the weighted ensemble averaging model
* Evaluate the ensemble model with the external dataset


---------------------------

### Enjoy!

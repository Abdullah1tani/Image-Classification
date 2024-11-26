# Image-Classification


## Project Structure

### `features` directory: 

- `features_extraction.py`: extract features from the cifar10 subset (5000 images for training, 1000 images for testing).
- `features_extraction_50d.py`: reduce features dimensions to 50x1.
- `features_utils`: functions used in other files in the `features` directory.

### `models` directory:
Contains 4 directories and 1 file. each directory contains files related to a model:

#### `CNN` directory:
- `CNN_implementation.py`: Classes and functions used to create, train, and test the CNN models in the other files.
- `CNN_train.py`: Trains and saves all CNNs models weights.
- `CNN_evaluation.py`: Displays metrics and confusion matrix of each CNN model.

#### `MLP` directory:
- `MLP_implementation.py`: Classes and functions used to create, train, and test the MLP models in the other files.
- `MLP_train.py`: Trains and saves all MLPs models weights.
- `MLP_evaluation.py`: Displays metrics and confusion matrix of each MLP model.

#### `Decision_Tree` directory:
- `DT_manual_implementation.py`: Manual decision tree class used to create, train, and test the decision tree model in the other files.
- `DT_train_test.py`: Trains and saves all decision tree models predictions.
- `DT_evaluation.py`: Displays metrics and confusion matrix of each decision tree model.

#### `Naive_Bayes` directory:
- `NB_manual_implementation.py`: Manual naive bayes class used to create, train, and test the naive bayes model in the other files.
- `NB_train_test.py`: Trains and saves all naive bayes models predictions.
- `NB_evaluation.py`: Displays metrics and confusion matrix of each naive bayes model.

`evaluation_utils.py`: functions used in the `models` directory and it's subdirectories.

## Installation Guide


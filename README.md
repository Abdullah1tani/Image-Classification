# Image-Classification

For this project, we ran all our code on Google Colab for faster computation but we organized the files in `.py` formats, you can find the output of the code in `code_output.ipynb`. 

In addition, we wrote a report for this project `image classification report.pdf`.

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
1. Clone the Github repository using this link: https://github.com/Abdullah1tani/Image-Classification.git
2. Download the models weights and predictions from the releases and unzip it.
3. Add every model's weight to its specific directory.

Note: Do not add an extra directory when adding the files, you need to remove the files from the directory from the zip file and add them to the specific directory. For example, when adding the file `features_extraction_50d.npz` to the project, you need to have the path to the file like this `features/features_extraction_50d.npz` and not `features/extracted_features/features_extraction_50d.npz`. You need to do that for all `.pth` and `.npz` files.

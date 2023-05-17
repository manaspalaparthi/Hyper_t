# Eye Classfication Model

The folder contains the code for the eye classification model built using tensorflow. 

we have trained 2 models on High resolution thermal data and low resolution thermal data.

Tha datasets are Private and cannot be shared.

brief description of the files in the folder:

- `eye_classification.py` : contains the code for the model and training
- `infarence.py` : contains the code for testing the models inference on a sample image
- `preprocessing.py` : contains the code for preprocessing the data to fit the model size requirements
- `video to images.py` : contains the code for converting video to images for training.


The model is trained on 2 classes:
- left eye
- right eye


## How to train the model on your own data

- create a folder called `data` in the root directory of the project
- inside the `data` folder create 2 folders `test` and `train`
- inside the `train` and `test` folders create 2 folders `left_eye` and `right_eye`
- inside the `left_eye` and `right_eye` folders put the images of the left and right eye respectively
- run the `preprocessing.py` script to resize the images to 224x224 and convert them to grayscale
- change the path to the data in the `eye_classification.py` script
- run the `eye_classification.py` script to train the model 


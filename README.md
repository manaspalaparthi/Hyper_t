# Hyper T

The repository contains the code for processing raw data for the project Project Hyper T.

Project has 4 categories of data:
- High resolution thermal data 
- Low resolution thermal data
- HRV (Heart Rate Variability) data
- core Pill temperature data


## Installation

### Requirements
- conda python 3.8
- pip
- tessract
- ffmpeg

```angular2html
conda create -n hyperT python=3.8
```

install the requirements.txt file

```angular2html
pip install -r requirements.txt
```
### These notebooks are used to view the data and do some basic EDA

```
├── notebooks
│   ├── core_pill_EDA.ipynb
│   ├── High_res_infrared_preprocessing.ipynb
│   ├── Low_res_video_EDA.ipynb
│   ├── HVR eda.ipynb

``` 
### Scripts inside the util folder are used to process the data

```
├── util
│   ├── Hrv_data_parsing.py
│   ├── high_res_data_parsing.py
│   ├── low_res_vide0_parsing.py

```

### Eye classification model

```
├── eye_classification
│   ├── eye_classification.py # script to train the model
│   ├── inference.py # script to run inference on the model
│   ├── preprocessing.py # script to preprocess the data to fit the model size requirements
│   ├── video to images.py # script to convert video to images for training. 

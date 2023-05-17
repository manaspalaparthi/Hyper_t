# Thermal Data Preprocessing  High Resolution / Low Resolution

The folder contains the code for processing the thermal video data processing using OCR and eye classification model.

The process for High resoluction and low resolution data is the same, the only difference is the eye classification model and the OCR location. 

# How it works

- The script `high_res_data_extration.py` takes the paths to the following:

    - `data_location` : path to the video file
    - `model` : path to the eye classification model
    - `output_path` : path to the output folder // default is `data_location` folder
  
- The script does the following:

the main loop reads all the participants as a list, each participant has a list of videos files as shown in the structure

Folder structure:

```
├── data
│   ├── participant_1
│   │   ├── video_1
│   │   ├── video_2
│   │   ├── video_3

```
After selecting a video, each frame is read and passed to the eye classification model to classify the eye in the frame,  the frame is passed to the OCR model to extract the temperature from the frame. 

At the end of the last frame, the script will save the extracted temperatures, eye side and frame number in a csv file in the output folder. 
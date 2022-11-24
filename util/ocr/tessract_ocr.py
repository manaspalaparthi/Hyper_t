import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pytesseract
import re
import datetime

plt.rcParams['figure.figsize'] = [10, 10]

def load_frame(frame_number, cap):
    cap.set(1, frame_number-1)
    res, frame = cap.read()
    return(frame)

def preprocess_frame(img):
    img = img[170:205,150:240,::-1]
    #img = cv2.resize(img,(600,400))
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return(img)


def extract_angle_confidence(result_dict):
    temp = np.NaN
    confidence = np.NaN
    for i in range(0, len(result_dict['text'])):
        confidence = int(result_dict['conf'][i])
        if confidence > 0:
            text = result_dict['text'][i]
            text = re.sub("[^0-9.^]", "", text)
            if len(text) > 0:
                temp = text

    return (temp, confidence)


cap = cv2.VideoCapture('../../HYP_T_Data_Files/HYP_T_12/Low_Res_Thermal_Camera/20221028T105933.MP4')
tesseract_config = r'--oem 3 --psm 13'
# img = load_frame(0,cap)
# img = preprocess_frame(img)
# result_dict = pytesseract.image_to_data(img, config = tesseract_config, output_type = pytesseract.Output.DICT)

angles = []
confidences = []
frames = []

count = 1
while cap.isOpened():
    res, frame = cap.read()
    if res == True:
        count =count +1
        img = preprocess_frame(frame)
        result_dict = pytesseract.image_to_data(img, config=tesseract_config, output_type=pytesseract.Output.DICT)
        angle, confidence = extract_angle_confidence(result_dict)
        print(f"temp {angle}")
        frames.append(count)
        angles.append(angle)
        confidences.append(confidence)
        # Press Q on keyboard to exit
        cv2.imshow('Frame', frame)
        cv2.imshow('"temp', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        cv2.destroyAllWindows()
        break

dict = {"frame_number": frames,"temp": angles}

df = pd.DataFrame(dict)

df.to_csv('20221028T105933.csv')

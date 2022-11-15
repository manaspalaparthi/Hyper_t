import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pytesseract
import re

plt.rcParams['figure.figsize'] = [10, 10]

def load_frame(frame_number, cap):
    cap.set(1, frame_number-1)
    res, frame = cap.read()
    return(frame)

def preprocess_frame(img):
    img = img[10:50,610:670,::-1]
    img = cv2.resize(img,(600,400))
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return(img)


def extract_angle_confidence(result_dict):
    angle = np.NaN
    confidence = np.NaN
    for i in range(0, len(result_dict['text'])):
        confidence = int(result_dict['conf'][i])
        if confidence > 0:
            text = result_dict['text'][i]
            text = re.sub("[^0-9^]", "", text)
            if len(text) > 0:
                angle = int(text)

    return (angle, confidence)


cap = cv2.VideoCapture('../../HYP_T_Data Files/HYP_T_12/Low Res Thermal Camera/20221028T105933.MP4')
tesseract_config = r'--oem 3 --psm 13'
img = load_frame(0,cap)
result_dict = pytesseract.image_to_data(img, config = tesseract_config, output_type = pytesseract.Output.DICT)

# angles = []
# confidences = []
# for i in range(0, 5_000):
#     img = load_frame(i, cap)
#     img = preprocess_frame(img)
#
#     tesseract_config = r'--oem 3 --psm 13'
#     result_dict = pytesseract.image_to_data(img, config=tesseract_config, output_type=pytesseract.Output.DICT)
#
#     angle, confidence = extract_angle_confidence(result_dict)
#
#     angles.append(angle)
#     confidences.append(confidence)
#
#
# angles = np.array(angles)
# confidences = np.array(confidences)
#
# np.save('angles.npy',angles)
# np.save('confidences.npy',confidences)


plt.imshow(img[:,:,::-1])
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pytesseract
import re
import tensorflow as tf
import datetime
from eye_classification.preprocessing import center_image ,resize_image
import os

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


#main function

if __name__ == "__main__":

    label_dict = {0: 'right', 1: 'left'}

    tesseract_config = r'--oem 3 --psm 13'

    model = tf.keras.models.load_model("../eye_classification/eye_classification_model2")

    location = "../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/Low Res Camera Recordings"

    # all the folders in the location

    folders = os.listdir(location)

    for folder in folders:
        print("participent name = ",folder)
        # all the videos in the folder end with .MP4
        videos = [video for video in os.listdir(location + "/" + folder) if video.endswith(".MP4")]
        for video in videos:
            print("video file name = ",video)
            cap = cv2.VideoCapture(location + "/" + folder + "/" + video)

            angles = []
            confidences = []
            frames = []
            labels = []

            count = 1
            while cap.isOpened():

                #frame rate is 8.fps
                res, frame = cap.read()
                if res == True:
                    count =count +1
                    img = preprocess_frame(frame)
                    result_dict = pytesseract.image_to_data(img, config=tesseract_config, output_type=pytesseract.Output.DICT)

                    img = center_image(frame)
                    # resize image
                    img = resize_image(img)

                    #classify
                    img = np.expand_dims(img, axis=0)

                    print("image shape",img.shape)

                    prediction = model.predict(img)

                    #add label to prediction

                    print(round(prediction[0][0]))

                    prediction = label_dict[round(prediction[0][0])]

                    # if prediction[0][0] == 0.0:
                    #     prediction = "right"
                    # else:
                    #     prediction = "left"

                    angle, confidence = extract_angle_confidence(result_dict)
                    angles.append(angle)
                    confidences.append(confidence)
                    frames.append(count)
                    labels.append(prediction)
                else:
                    break

            # save data to csv

            df = pd.DataFrame({"frame_number": frames,"temp": angles, "side": labels})
            df.to_csv(location + "/" + folder + "/" + video +".csv", index = False)
            print("file saved",location + "/" + folder + "/" + video +".csv")
            cap.release()
            cv2.destroyAllWindows()


    # cap = cv2.VideoCapture(0)
    # tesseract_config = r'--oem 3 --psm 13'
    # # img = load_frame(0,cap)
    # # img = preprocess_frame(img)
    # # result_dict = pytesseract.image_to_data(img, config = tesseract_config, output_type = pytesseract.Output.DICT)
    #
    # # load eye_classification saved model
    # model = tf.keras.models.load_model("../../eye_classification/eye_classification_model")
    #
    #
    #
    # angles = []
    # confidences = []
    # frames = []
    # labels = []
    #
    # count = 1
    # while cap.isOpened():
    #
    #     #frame rate is 8.fps
    #     res, frame = cap.read()
    #     if res == True:
    #         count =count +1
    #         img = preprocess_frame(frame)
    #         result_dict = pytesseract.image_to_data(img, config=tesseract_config, output_type=pytesseract.Output.DICT)
    #
    #         img = center_image(frame)
    #         # resize image
    #         img = resize_image(img)
    #
    #         #classify
    #         img = np.expand_dims(img, axis=0)
    #
    #         prediction = model.predict(img)
    #
    #         #add label to prediction
    #
    #         prediction = label_dict[prediction[0][0]]
    #         print(f"prediction {prediction}")
    #
    #         angle, confidence = extract_angle_confidence(result_dict)
    #         print(f"temp {angle}")
    #         frames.append(count)
    #         angles.append(angle)
    #         labels.append(prediction)
    #
    #         cv2.imshow('Frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             cv2.destroyAllWindows()
    #             break
    #     else:
    #         cv2.destroyAllWindows()
    #         break
    #
    # dict = {"frame_number": frames,"temp": angles, "side": labels}
    #
    # df = pd.DataFrame(dict)
    #
    # df.to_csv(location+file_name+".csv")

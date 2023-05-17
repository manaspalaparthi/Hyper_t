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
    img = img[150:190,570:630,::-1]
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

    label_dict = {0: 'left', 1: 'other', 2: 'right'}

    tesseract_config = r'--oem 3 --psm 13'

    # location of the folder that containes all the participant videos.
    data_location = "../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/High Res Thermal Camera/video_files/"

    # load model for classification
    model = tf.keras.models.load_model("../eye_classification/eye_classification_model_high_res4")

    # all the folders in the location end with "_"
    folders = [folder for folder in os.listdir(data_location) if folder.endswith("_")]

    for folder in folders:
        print("participent name = ", folder)
        # all the videos in the folder end with .MP4
        videos = [video for video in os.listdir(data_location + "/" + folder) if video.endswith(".avi")]
        for video in videos:
            print("video file name = ", video)
            cap = cv2.VideoCapture(data_location + "/" + folder + "/" + video)

            angles = []
            confidences = []
            frames = []
            labels = []

            count = 1
            while cap.isOpened():
                # frame rate is 8.fps
                res, frame = cap.read()
                if res == True:
                    count = count + 1
                    img = preprocess_frame(frame)
                    result_dict = pytesseract.image_to_data(img, config=tesseract_config,
                                                            output_type=pytesseract.Output.DICT)

                    #img = center_image(frame)
                    # resize image
                    image = resize_image(frame)

                    # bgr to rgb

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    image = image.astype("float32") / 255.0

                    # classify
                    img = np.expand_dims(image, axis=0)

                    print("image shape", img.shape)

                    prediction = model.predict(img)

                    print("prediction = ", prediction)

                    label = np.argmax(prediction)

                    print("label = ", label_dict[label])

                    labels.append(label_dict[label])

                    angle, confidence = extract_angle_confidence(result_dict)

                    print("angle = ", angle, "confidence = ", confidence)
                    angles.append(angle)
                    confidences.append(confidence)
                    frames.append(count)
                    cv2.imshow('Frame2', frame)
                    cv2.imshow('Frame', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                else:
                    cv2.destroyAllWindows()
                    break

            # save data to csv
            cap.release()
            cv2.destroyAllWindows()

            df = pd.DataFrame({"frame_number": frames, "temp": angles, "confidence": confidences, "eye_side": labels})
            df.to_csv(data_location + "/" + folder + "/" + video + ".csv", index=False)
            print("file saved", data_location + "/" + folder + "/" + video + ".csv")
            cap.release()
            cv2.destroyAllWindows()

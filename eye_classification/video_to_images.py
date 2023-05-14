import numpy as np
import cv2
import os
import datetime
import random

#video to frames
file_name = "../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/High Res Thermal Camera/video_files/h18/rec_0002.avi"

out_dir = "highres_data/frames/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

cap = cv2.VideoCapture(file_name)
count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        #save frame as JPEG file name only time stamp
        print(out_dir + str(count) + str(datetime.datetime.now()) +  ".jpg")
        cv2.imwrite(out_dir + str(count)+ str(random.random()) +".jpg", frame)
        count += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()








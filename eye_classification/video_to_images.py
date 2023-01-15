import numpy as np
import cv2
import os
import datetime

#video to frames
file_name = "../HYP_T_Data_Files/HYP_T_08/20221026T112650.MP4"

out_dir = "frames/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

cap = cv2.VideoCapture(file_name)
count = 0
while cap.isOpened():
    ret, frame = cap.read()


    if ret == True:
        #save frame as JPEG file name only time stamp
        print(f"frame {count}")
        cv2.imwrite(out_dir + str(count) + "8008.jpg", frame)
        count += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()








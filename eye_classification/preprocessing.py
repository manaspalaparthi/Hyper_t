import numpy as np
import cv2
import os



#center image to square
def center_image(img):
    height, width = img.shape[:2]
    if height == width:
        square_img = img
    elif height > width:
        diff = height - width
        diff = diff // 2
        square_img = img[diff:height-diff, 0:width]
    else:
        diff = width - height
        diff = diff // 2
        square_img = img[0:height, diff:width-diff]
    return square_img

#resize image to 400x400

def resize_image(img):
    dim = (200, 200)
    resized = cv2.resize(img, dim)
    return resized

#main

if __name__ == "__main__":
    #load all image from folder test
    path = "train/left/"
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        #center image
        img = center_image(img)
        #resize image
        img = resize_image(img)
        print(f"image {filename} shape {img.shape}")
        #save image
        cv2.imwrite("train_400/left/" + filename, img)







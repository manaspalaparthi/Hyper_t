import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

label_dict = {0: 'left', 1: 'other', 2: 'right'}

model = tf.keras.models.load_model("eye_classification_model_high_res4")

#load image using cv2

img = cv2.imread("highres_data/test/left/6020.8304223220253452.jpg")

#preprocess image

image = cv2.resize(img,(200,200))

#rescale=1.0 / 255.0

image = image.astype("float32") / 255.0

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#predict image

img = np.expand_dims(image, axis=0)

prediction = model.predict(img)

label = np.argmax(prediction)

print(prediction[0])
print("label = ", label_dict[label])

cv2.imshow('image', image)

cv2.waitKey(0)


from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO('./runs/classify/train7/weights/last.pt')  

image_path = input("Enter the path of the image: ")


results = model(image_path)

names_dict = results[0].names

probs = results[0].probs.data.tolist()
#print(results)
print(names_dict)
print("here is probs: " , probs)

print(names_dict[np.argmax(probs)])
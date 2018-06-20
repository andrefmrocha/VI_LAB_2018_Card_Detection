import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def open_images():
    imlist = sorted(os.listdir("./NJUID_Cropped"))
    x_Card = np.array([cv2.imread("./NJUID_Cropped/" + img) for img in imlist if "card" in img])
    x_Camera = np.array([cv2.imread("./NJUID_Cropped/" + img) for img in imlist if "camera" in img])
    return x_Card, x_Camera
# print(len(list_labels))
# prop - Proportion of negatives
# random_list = random.sample(range(3), 2)
# print(random_list)
def gen_data(card, camera, prop = 1):
    list_card = list(card/255)
    list_camera = list(camera/255)
    y = [1] * len(list_card)
    y.extend([0] * (len(y) * prop))
    i = 0
    while(i < len(card)):
        random_list = random.sample(range(len(list_card)), prop+1)
        list_card.append(list_card[i] * prop)
        for j in random_list:
            if(j != i and len(list_camera) != len(list_card)):
                list_camera.append(list_camera[j])
        i+=1

    return list_card,list_camera,y
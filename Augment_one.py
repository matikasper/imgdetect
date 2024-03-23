import os
import cv2
import albumentations as alb
from matplotlib import pyplot as plt
import numpy as np
import json

augmentor = alb.Compose([alb.RandomCrop(width=4000, height=1800),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='albumentations',
                                                   label_fields=['class_labels']))

img = cv2.imread(os.path.join('D:\\Magisterka\\ProgramNaMagisterke\\data\\test\\images\\zdjecie19.jpg'))

with open(os.path.join('D:\\Magisterka\\ProgramNaMagisterke\\data\\test\\labels\\zdjecie19.json'), 'r') as f:
    label = json.load(f)

coords = [0, 0, 0, 0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords = list(np.divide(coords, [4000, 1800, 4000, 1800]))

augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [3000, 1000]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [3000, 1000]).astype(int)),
              (255, 0, 0), 2)

plt.imshow(augmented['image'])
plt.show()
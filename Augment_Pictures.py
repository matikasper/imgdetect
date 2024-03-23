import os
import cv2
import tensorflow as tf
import albumentations as alb
import numpy as np
import json


def limit_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def augment_all_pictures():
    augmentor = alb.Compose([alb.RandomCrop(width=3000, height=1200),
                             alb.HorizontalFlip(p=0.5),
                             alb.RandomBrightnessContrast(p=0.2),
                             alb.RandomGamma(p=0.2),
                             alb.RGBShift(p=0.2),
                             alb.VerticalFlip(p=0.5)],
                            bbox_params=alb.BboxParams(format='albumentations',
                                                       label_fields=['class_labels']))

    for image in os.listdir(os.path.join('D:\\Magisterka\\ProgramNaMagisterke\\data\\test\\images')):
        img = cv2.imread(os.path.join('D:\\Magisterka\\ProgramNaMagisterke\\data\\test\\images', image))

        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join('D:\\Magisterka\\ProgramNaMagisterke\\data\\test\\labels',
                                  f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[0] = label['shapes'][0]['points'][0][1]
            coords[0] = label['shapes'][0]['points'][1][0]
            coords[0] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [4000, 1800, 4000, 1800]))

            try:
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['napoina'])
                    cv2.imwrite(os.path.join('D:\\Magisterka\\ProgramNaMagisterke\\aug_data\\test\\images',
                                             f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                    annotation = {'image': image}

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0

                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1

                    else:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0

                    with open(os.path.join('D:\\Magisterka\\ProgramNaMagisterke\\aug_data\\test\\labels',
                                           f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation, f)


            except Exception as e:
                print('eeee')


augment_all_pictures()

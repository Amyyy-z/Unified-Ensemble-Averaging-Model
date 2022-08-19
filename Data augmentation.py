#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import glob
import datetime
from PIL import Image
from PIL import ImageEnhance

image_list = []
files = glob.glob(r'C:/.../0/*.jpg') #identify the original image folder path

for filename in files:
    image = cv2.imread(filename)
    image_list.append(image)
    
print(np.asarray(image_list).shape)

orig = np.asarray(image_list)

#demonstrate the original images
plt.figure(figsize=(9,9))
i = 0
for img in orig[40:56]:
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    i += 1
plt.suptitle("Original", fontsize=20)
plt.show()

#augment images through rotation
rotate = [Image.fromarray(img, 'RGB').rotate(np.random.choice([270])) for img in orig] #the numbers can be changes accordingly 90, 180, 270

#change to flip if want to use flip method for augmentation
flip = [cv2.flip(img, 1) for img in orig]

#demonstrate same batch of images again after rotation
plt.figure(figsize=(9,9))
i = 0
for img in rotate[40:56]:
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    i += 1
plt.suptitle("Rotate", fontsize=20)
plt.show()

#create a new path for augmented images
out_folder_path = r'C:/.../augmented/'

image_no = 0

for image in rotate:
    name = r'C:/.../augmented/' + '270_' + str(image_no) + '.png'
    image.save(name)
    image_no += 1

import numpy as np
import os
import math
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io as io

def down_sampling(image):  
    image = np.array(image)
    h, w= image.shape[0],image.shape[1]
    image = Image.fromarray(image)
    image = image.resize((int(h/2),int(w/2)),Image.ANTIALIAS)  
    return image

def read_path(path_name = None,label_name = None):
    label_index = 0
    images = []
    labels = []
    path_list = os.listdir(path_name)
    path_list.sort(key = lambda x: int(x[:-4]))  
    for dir_item in path_list:
            full_path = os.path.abspath(os.path.join(path_name,dir_item))
            if os.path.isdir(full_path):
                read_path(full_path,None)
            else:
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
                    image = down_sampling(image)
                    image = np.array(image)
                else:
                    break
                images.append(image)
    
    path_list1 =os.listdir(label_name)  
    path_list1.sort(key = lambda x: int(x[:-4]))
    for dir_item1 in path_list1:
            full_path1 = os.path.abspath(os.path.join(label_name,dir_item1))  
            if os.path.isdir(full_path1):
                read_path(None,full_path1)
            else:
                if dir_item.endswith('.jpg'):
                    label = cv2.imread(full_path1,cv2.IMREAD_GRAYSCALE)
                    label = segmentation(label)
                    label = down_sampling(label)
                    label = np.array(label)
                else:
                    break  
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images,labels

def segmentation(image):
    _, image = cv2.threshold(image,0,1,cv2.THRESH_BINARY)
    return image

def load_dataset(path_name,label_name):   
    images, labels = read_path(path_name,label_name)
    images = np.array(images)
    return images,labels

def saveResult(save_path1,save_path2,npyfile):
    for i,item in enumerate(npyfile,start = 1):
        img_eng = item[:,:,0]
        img_num = item[:,:,1]

def combine_images(images1,images2):
    num = images1.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = images1.shape[1:3]
    number = np.zeros((height*shape[0], width*shape[1]),
                        dtype=images1.dtype)
    letter = np.zeros((height*shape[0], width*shape[1]),
                        dtype=images1.dtype)

    for index, img in enumerate(images1):
        i = int(index/width)
        j = index % width
        letter[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]

    for index, img in enumerate(images2):
        i = int(index/width)
        j = index % width
        number[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return letter,number

if __name__ == "__main__":
    path_name = ''
    label_name =''
    
    load_dataset(path_name,label_name)

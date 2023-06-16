# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:13:38 2021

@author: Mahsa
"""
# MLP = WLD + LBP
# 1. WLD: Weber Local Descriptor
# simulates the human perception mechanism for the surroundings to describe the visual clues in the image,
# producing a descriptor encoding both differential excitation and orientation. On the other hand, many
# learning based descriptors

import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
# from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from skimage import io, exposure
import os
from PIL import Image
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
# from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from skimage import io, exposure
import os
from PIL import Image
from scipy.spatial.distance import euclidean
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
import joblib

import cv2
import numpy as np
import sys
# from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from skimage import io, exposure
import os
from math import atan
from PIL import Image
from scipy.spatial.distance import euclidean


def weber(grayscale_image):
    grayscale_image = grayscale_image.astype(np.float64)
    grayscale_image[grayscale_image==0] = np.finfo(float).eps
    neighbours_filter = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ])
    convolved = convolve2d(grayscale_image,neighbours_filter, mode='same')
    weber_descriptor = convolved-8*grayscale_image
    weber_descriptor = weber_descriptor/grayscale_image
    weber_descriptor = np.arctan(weber_descriptor)
    return weber_descriptor

def mb_lbp_histogram(color_image):
    img = color.rgb2gray(color_image)
    patterns = multiblock_lbp(img, 8, 1, 10,10)
    hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
    return patterns

def lbp_histogram(color_image):
    img = color.rgb2gray(color_image)
    patterns = local_binary_pattern(img, 8, 1)
    hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
    return patterns

def bgr2rgb(img):
    b,g,r = cv2.split(img)
    return cv2.merge([r,g,b])

def DarkChannel(I, w = 15):
    M, N = I.shape[0:2]
    J_dark = np.zeros((M, N))
    w_pad = int(w/2)
    pad = np.pad(I, ((w_pad, w_pad), (w_pad, w_pad), (0, 0)), mode = 'edge')
    for i, j in np.ndindex(J_dark.shape):
        J_dark[i, j] = np.min(pad[i:i+w, j:j+w, :])
    return J_dark

def AtmLight(I, J_dark, p = 0.001):
    M, N = J_dark.shape
    I_flat = I.reshape(M*N, 3)
    dark_flat = J_dark.ravel()
    idx = (-dark_flat).argsort()[:int(M*N*p)]
    arr = np.take(I_flat, idx, axis = 0)
    A = np.mean(arr, axis = 0)
    return A.reshape(1,3)

def TransmissionEstimate(I, A, w = 15, omega = 0.95):
    return 1 - omega*DarkChannel(I/A, w)

def Guidedfilter(im, p, r = 200,eps = 1e-06):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def Recover(im, t, A, t0 = 0.1):
    rec = np.zeros(im.shape)
    t = cv2.max(t,t0)

    for ind in range(0,3):
        rec[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return rec


def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(abs(diff))  # Manhattan norm
    return m_norm

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr
    
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

import csv
img_path = r"C:\Users\Mahsa\Desktop\Tasks\7-#P3105-CV-Matlab\dataset\train\train1" # Enter Directory of all images  

folder = img_path
images = [os.path.join(root, filename)
          for root, dirs, files in os.walk(folder)
          for filename in files
          if filename.lower().endswith('.jpg')]

img_path_test = r"C:\Users\Mahsa\Desktop\Tasks\7-#P3105-CV-Matlab\dataset\train\test" # Enter Directory of all images  

folder_test = img_path_test
images_test = [os.path.join(root, filename)
          for root, dirs, files in os.walk(folder_test)
          for filename in files
          if filename.lower().endswith('.jpg')]

import itertools
distance_pred = []
count=0
train_labels=[]
train_features1=[]
train_features2=[]
train_features=[]
for image in itertools.islice(images , 0, 15):
    image_name=image.split("\\")[-1]
    train_labels.append(image_name)

    src = cv2.imread(image)
    I = src.astype('float64')/255
    src_gray_read = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = np.float64(src_gray_read)/255
    src2 = cv2.imread(image,0)
    weber_=weber(src2)
    
    lbp_=lbp_histogram(src)

    prediction1 =lbp_
    prediction2 =weber_
    dark = DarkChannel(I)
    A = AtmLight(I, dark)
    et = TransmissionEstimate(I, A)
    t = Guidedfilter(src_gray, et)
    J = Recover(I, t, A)
    prediction =J
    train_features.append(prediction)
    train_features1.append(prediction1)
    train_features2.append(prediction2)

import matplotlib.pyplot as plt    
train=[]
train=list(zip(train_labels, train_features))  
distance_labels = [] 
distance_image = []
for image_test in images_test:
    src_test = cv2.imread(image_test)
    img1=src_test
    for train_labels,train_feature in train:
        img2 = train_feature
        # img2 =  np.expand_dims(img2, axis=2)
        n_m= compare_images(img1, img2)
        distance_pred.append(n_m)
        distance_image.append(img2)
  
        
min_hist=min(distance_pred)  
min_hist_index=distance_pred.index(min(distance_pred))
# print("{:.2f}".format(round(min_hist, 2)))
# print(min_hist)

image_pred=distance_image[min_hist_index]


cv2.imshow('I', image_pred)
cv2.waitKey(0)


plt.hist(src.ravel(),256,[0,256])
plt.show()

a = np.array(distance_pred)
mean_a=np.mean(a, axis=0)
dist_a=round(min_hist, 2)
std_a= np.std(a, axis=0)
standardize_dis=((dist_a-mean_a)/std_a)
# print(standardize_dis)



def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

v=[343081.31, 313081.31, 283081.31]
b=normalize(v)
# print(b)

aa=[]
with open('mb_lmp_sample_40.csv', 'w', encoding='UTF8') as f1:
    csvcreator_x = csv.writer(f1)
    # avg_acc = statistics.mean(distance_pred)  # Average distances of all images  
    # print(avg_acc)
    for ii in distance_pred:
        aa.append(atan(ii)/(len(distance_pred)-1))
    csvcreator_x.writerow(aa)
    
f1.close()
    
import statistics
bb=[]
avg_acc = statistics.mean(distance_pred)  # Average distances of all images
# print("Average distance %d" %(avg_acc*(1/count)))

float_list = [count,avg_acc]
with open('mb_lmp_sample_x.csv', 'a', newline='', encoding='UTF8') as f_wrapup:
    csvcreator_x = csv.writer(f_wrapup)
    for ii in float_list:
        bb.append(atan(ii)/(len(distance_pred)-1))   
    csvcreator_x.writerow(bb)
f_wrapup.close()
print(bb)
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

f=open('data_encoder.pkl','rb')
data=pickle.load(f)
img=np.zeros([396*396,1]).astype(np.uint8)
index=0
for d in data:
    '''if d[0]>d[1] and d[0]>d[2] and d[0]>d[3]:
        img[index]=[0,0,0]
    elif d[1]>[0] and d[1]>d[2] and d[1]>d[3]:
        img[index]=[60,60,60]
    elif d[2]>d[0] and d[2]>d[1] and d[2]>d[3]:
        img[index]=[120,120,120]
    elif d[3]>d[0] and d[3]>d[1] and d[3]>d[2]:
        img[index] = [255, 255, 255]'''
    if d[0]>d[1]:
        img[index]=255
    else:
        img[index]=0

    index+=1
img=img.reshape(396,396 )
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.imwrite('lena_filter.jpg',img)


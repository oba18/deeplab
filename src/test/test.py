import numpy as np
import scipy.io
# from matplotlib import pyplot as plt
import cv2


file_name = '2008_003228'

data = scipy.io.loadmat(f'../../trainval/Annotations_Part/{file_name}.mat')

d = data['anno'][0][0][1][:, 1][0][3]
print(d)
print(type(d))


img_path = f'../../VOC2010/JPEGImages/{file_name}.jpg'
print(img_path)
cv2.imshow(f'{file_name}', img_path)
cv2.waitKey()
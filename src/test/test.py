import numpy as np
import scipy.io
from matplotlib import pyplot as plt

data = scipy.io.loadmat('/Users/masashi/Desktop/trainval/Annotations_Part/2008_003228.mat')

# a = data['anno'][0][0][1][:, 1][0][3][:, 1][0][1]
# print(a)

# print(np.array([[1]]).shape)


a = data['anno'][0][0][1]
t = type(a)
s = a.shape
print(a)
print(t)
print(s)

# plt.imshow(a)
# plt.show()
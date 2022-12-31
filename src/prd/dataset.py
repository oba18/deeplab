from torch.utils.data import Dataset
import pathlib
import os
import numpy as np
from PIL import Image, ImageFile
import scipy
import cv2

class MyDataset(Dataset):
  
  def __init__(
    self,
    ids,
    images_dir,
    masks_dir,
    augmentation=None,
    preprocessing=None
  ):

    self.classes = [
    'background',
    'hair',
    'head',
    'lear',
    'leye',
    'lebrow',
    'lfoot',
    'lhand',
    'llarm',
    'llleg',
    'luarm',
    'luleg',
    'mouth',
    'neck',
    'nose',
    'rear',
    'reye',
    'rebrow',
    'rfoot',
    'rhand',
    'rlarm',
    'rlleg',
    'ruarm',
    'ruleg',
    'torso'
  ]

    self.ids = ids
    self.image_fps = [os.path.join(images_dir, image_id)+".jpg" for image_id in self.ids]
    self.masks_fps = [os.path.join(masks_dir, image_id)+".mat" for image_id in self.ids]
    self.class_values = [self.classes.index(cls.lower()) for cls in self.classes]

    self.augmentation = augmentation
    self.preprocessing = preprocessing
    self.Onhot = True
    
  def __getitem__(self, i):
    image = np.asarray(Image.open(self.image_fps[i]))
    shape = image.shape
    mask = np.zeros((shape[0], shape[1], len(self.classes)))
    background = np.ones((shape[0], shape[1]))
    data = scipy.io.loadmat(self.masks_fps[i])
    target = data['anno'][0][0][1][0]
    for t in target:
      if t[0][0] == 'person':
        background = background - t[2]
        mask[:, :, 0] = background
        np.place(mask[:, :, 0], mask[:, :, 0] < 0, 0)
        for a in t[3]:
          for b in a:
            mask[:, :, self.classes.index(b[0][0])] = mask[:, :, self.classes.index(b[0][0])] + b[1]

    if self.augmentation:
      sample = self.augmentation(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']

    # apply preprocessing
    if self.preprocessing:
      sample = self.preprocessing(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    
    # return  cv2.resize(image, self.size), cv2.resize(mask, self.size)
    return  image, mask
  
  def __len__(self):
    return len(self.ids)
  
  
import numpy as np
import torch
import cv2
import dataset
from sklearn.model_selection import train_test_split
import pathlib
import albumentations as albu
from torch.utils.data import DataLoader



# テンソル化
def to_tensor(x, **kwargs):
    return x.transpose(2, 1, 0).astype('float32')

def to_norm(x ,  **kwargs):
    return  x/255

def get_preprocessing(preprocessing_fn):
    # train/testに関わらず加えるデータ加工
    if preprocessing_fn == None:
      _transform = [
            albu.Lambda(image=to_norm),
            albu.Resize(256,256),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
      _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Resize(256,256),
            albu.Lambda(image=to_tensor, mask=to_tensor),
          ]
    return albu.Compose(_transform)
  

device = 'cpu'
predict_model = torch.load('./DeepLabV3Plus_efficientnet-b3.pth', map_location=torch.device(device))
# predict = predict_model(image_torch.to(device))

files = pathlib.Path('../../persons_dataset/mat/').glob('*.mat')
files_list = []

# 全ファイル取得
for f in files:
  files_list += [f.name[:-4]]

ids = train_test_split(files_list)

TRAIN = 0
VALID = 1
classes = [
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

valid_dataset = dataset.MyDataset(
  ids[VALID],
  '../../persons_dataset/jpg',
  '../../persons_dataset/mat',
  preprocessing=get_preprocessing(None),
)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last = True, num_workers=0)

for i , (images, label) in enumerate(valid_loader):
  predict_model.eval()
  predict = predict_model(images.to(device))
  print(predict.shape)
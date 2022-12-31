import dataset
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils
import albumentations as albu

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
files = pathlib.Path('../../persons_dataset/mat/').glob('*.mat')
files_list = []

# 全ファイル取得
for f in files:
  files_list += [f.name[:-4]]

# モデルを宣言
ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'

ACTIVATION = "softmax2d"

DECODER = "DeepLabV3Plus"
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(classes),
    activation=ACTIVATION,
)

model = model.to("cpu")

preprocessing_fn =smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
ids = train_test_split(files_list)

train_dataset = dataset.MyDataset(
  ids[TRAIN],
  '../../persons_dataset/jpg',
  '../../persons_dataset/mat',
  preprocessing=get_preprocessing(preprocessing_fn),
)

valid_dataset = dataset.MyDataset(
  ids[VALID],
  '../../persons_dataset/jpg',
  '../../persons_dataset/mat',
  preprocessing=get_preprocessing(preprocessing_fn),
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last = True, num_workers=0)


# 精度確認指標
metrics = [
    utils.metrics.IoU(threshold=0.5),
]
# loss  #mode='multilabel'は1枚画像の中に複数種類の物体が存在するときに使う
loss = smp.losses.DiceLoss(mode='multilabel')
loss.__name__ = 'dice_loss'

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),#あまりにここが大きいと全て0、学習に失敗します。
])

DEVICE = 'cpu'
# 1Epochトレイン用
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)
valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# 学習 
max_score = 0
for i in range(0, 30):

    print('\nEpoch: {}'.format(i))
    try:
        train_logs = train_epoch.run(train_loader)
        val_logs = valid_epoch.run(valid_loader)
        print(val_logs)
    except Exception as e:
        print(e)

    # do something (save model, change lr, etc.)
    if max_score < val_logs['iou_score']:
        max_score = val_logs['iou_score']
        torch.save(model, f'{DECODER}_{ENCODER}.pth')
        print('Model saved!')
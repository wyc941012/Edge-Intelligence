import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data

from data import get_data_set

test_x,test_y,test_l=get_data_set("test")
test_x=torch.from_numpy(test_x[0:100]).float()
test_y=torch.from_numpy(test_y[0:100]).long()

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_SAVE_PATH = "model/"

model=torch.load(_SAVE_PATH+"layermodel.pkl", map_location='cpu')
device = torch.device("cpu")
#model.to(device)

# 定义数据库
test_dataset = Data.TensorDataset(test_x, test_y)

# 定义数据加载器
test_loader = Data.DataLoader(dataset = test_dataset, batch_size = _BATCH_SIZE, shuffle = False)


correct_classified = 0
total = 0

for batch_num, (batch_xs, batch_ys) in enumerate(test_loader):
    outputs = model(batch_xs)
    prediction = torch.max(outputs.data, 1)
    total = total + batch_ys.size(0)
    correct_classified += np.sum(prediction[1].numpy() == batch_ys.numpy())

acc=(correct_classified/total)*100

print("Accuracy on Test-Set:{0:.2f}%({1}/{2})".format(acc,correct_classified,total))
#########################加载数据##############################
import torch
import torchvision
import os
import pandas as pd

def read_data(is_train=True):
    """读取检测数据集中的图像和标签"""
    data_dir = 'detection/'
    csv_fname = os.path.join(data_dir, 'sysu_train' if is_train
                             else 'sysu_val')

    csv_data = pd.read_csv(csv_fname+'/label.csv')
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        tem = os.path.join(csv_fname, 'images', f'{img_name}')
        images.append(torchvision.io.read_image(tem))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


class Dataset(torch.utils.data.Dataset):
    """一个用于加载检测数据集的自定义数据集"""

    def __init__(self, is_train):
        self.features, self.labels = read_data(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
                                                   is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


def load_data(batch_size):
    """加载检测数据集"""
    train_iter = torch.utils.data.DataLoader(Dataset(is_train=True),
                                             batch_size, shuffle=True)
    # val_iter = torch.utils.data.DataLoader(Dataset(is_train=False),
    #                                       batch_size)
    return train_iter  # , val_iter



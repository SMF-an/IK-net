import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import json


def read_normalization_data(): # 读取归一化数据
    with open('data/normalization.json','r') as f:   #这里地址由data_generator函数保存
        normalization_data = json.load(f)
    pos_min = np.array(normalization_data["pos_min"])
    pos_max = np.array(normalization_data["pos_max"])
    rpy_min = np.array(normalization_data["rpy_min"])
    rpy_max = np.array(normalization_data["rpy_max"])
    return pos_min, pos_max, rpy_min, rpy_max


class IKDataset(Dataset):
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')
        self.results = self.data['results']  # 缓存结果数据集
        self.inputs = self.data['inputs']    # 缓存输入数据集
        self.pos_min, self.pos_max, self.rpy_min, self.rpy_max = read_normalization_data()
        self.pos_norm = self.pos_max - self.pos_min
        self.rpy_norm = self.rpy_max - self.rpy_min

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        positions = torch.Tensor(self.results[idx])
        joint_angles = torch.Tensor(self.inputs[idx])
        positions[:3] = (positions[:3] - torch.Tensor(self.pos_min)) / torch.Tensor(self.pos_norm)
        positions[3:] = (positions[3:] - torch.Tensor(self.rpy_min)) / torch.Tensor(self.rpy_norm)
        input_tensor = positions.squeeze(0)        
        return input_tensor, joint_angles

class IKDatasetVal(Dataset):
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')
        self.pos_min, self.pos_max, self.rpy_min, self.rpy_max = read_normalization_data()
        self.pos_norm = self.pos_max - self.pos_min
        self.rpy_norm = self.rpy_max - self.rpy_min

    def __len__(self):
        return len(self.data.get('results'))

    def __getitem__(self, idx):
        positions = torch.Tensor(self.data.get('results')[len(self.data.get('results')) - idx - 1])
        joint_angles = torch.Tensor(self.data.get('inputs')[len(self.data.get('results')) - idx - 1])
        positions[:3] = (positions[:3] - torch.Tensor(self.pos_min)) / torch.Tensor(self.pos_norm)
        positions[3:] = (positions[3:] - torch.Tensor(self.rpy_min)) / torch.Tensor(self.rpy_norm)
        input = positions.squeeze(0)
        return input, joint_angles


 



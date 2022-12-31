import torch
from torch.utils.data import Dataset, DataLoader

import glob
import os
import numpy as np

class EdfDataset(Dataset):
    def __init__(self, data_path, seq_len, is_train=True, shuffle_seed=42):
        super(EdfDataset,self).__init__()
        # 加载数据
        files = glob.glob(os.path.join(data_path, "*.npz"))
        # 随机抽选训练集和测试集样本
        np.random.seed(shuffle_seed)
        shuffle_idx = np.arange(len(files))
        np.random.shuffle(shuffle_idx)
        # 划分训练集和测试集
        train_idx = shuffle_idx[:int(len(files)*0.8)]
        test_idx = shuffle_idx[int(len(files)*0.8):]
        if is_train:
            nfiles = [f for i,f in enumerate(files) if i in train_idx]
        else:
            nfiles = [f for i,f in enumerate(files) if i in test_idx]
        # 读取数据并转换为tensor
        X_data = []
        y = []
        for fi in nfiles:
            data = np.load(fi)
            for seq_idx in range(len(data['x'])//seq_len):
                item_x = data['x'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_x = torch.from_numpy(item_x)
                item_y = torch.from_numpy(item_y)
                assert item_x.shape[0] == item_y.shape[0]
                X_data.append(item_x)
                y.append(item_y)
        
        assert len(X_data) == len(y)

        self.X_data = X_data
        self.y = y

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)



if __name__ == "__main__":
    data_path = "./data/sleepedf/npz"

    edf_dataset = EdfDataset(data_path, seq_len=20, is_train=True)
    edf_dataloader = DataLoader(edf_dataset, batch_size=15, shuffle=True)

    for X, y in edf_dataloader:
        print(X.shape, y.shape)

    # data_list = glob.glob(os.path.join(data_path,"*.npz"))

    # data_list.sort()

    # for fi in data_list:
    #     data = np.load(fi)
    #     print(data['x'].shape,data['y'].shape)

    # data = np.load(data_list[0])

    # print(data['x'].shape, data['y'].shape)
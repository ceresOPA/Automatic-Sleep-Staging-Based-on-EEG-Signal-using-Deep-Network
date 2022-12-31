from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import torch

import sklearn.metrics as skmetrics
import numpy as np
import timeit
import matplotlib.pyplot as plt

from dataset import EdfDataset
from network import SleepNet
from focal_loss import FocalLoss

import argparse

# wandb用于在线追溯实验，方便实验结果保存和调参，如若需要解开注释即可
# import wandb

# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--network", type=str, default="GRU", help="GRU | LSTM | Attention")
args = parser.parse_args()

#定义超参数
n_epochs = args.n_epochs # 迭代次数,每个epoch会对整个训练集遍历一遍
batch_size = args.batch_size # 一次加载的数据量，对一个epoch中的样本数的拆分
learning_rate = 0.001 # 学习率，或者说步长
seq_len = args.seq_len
network = args.network

# wandb.init(
#     project="sleepstaging",
    
#     config={
#     "learning_rate": learning_rate,
#     "n_epochs": n_epochs,
#     "batch_size": batch_size,
#     "seq_len": seq_len,
#     "network": "GRU&FocalLoss@75",
#     }
# )

#加载数据
data_path = "./data/sleepedf/npz"
train_data = EdfDataset(data_path, seq_len=seq_len, is_train=False)
#按8:2划分训练集和验证集
train_size = int(len(train_data)*0.8)
validate_size = len(train_data) - train_size
train_dataset,validate_dataset = random_split(train_data,[train_size,validate_size])
#使用DataLoader加载数据集，转换为迭代器
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
validate_dataloader = DataLoader(validate_dataset,batch_size=batch_size) #用于训练中验证模型效果，进而可以动态调整超参数，控制训练

#设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

#加载模型
model = SleepNet(network=network, seq_len=seq_len)
# model.load_state_dict(torch.load("./model_encoder.pt"))
model.to(device)

#设置优化器
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

#设置损失函数
criterion = FocalLoss(gamma=0.75)

best_acc = -1

#训练模型
for epoch_idx in range(n_epochs):
    start = timeit.default_timer()
    model.train()
    train_loss = []
    train_trues = []
    train_preds = []
    for batch_idx, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.reshape(-1,).to(device, dtype=torch.long)
        pred = model(X)
        loss = criterion(pred, y)
        train_trues.append(y.cpu())
        train_preds.append(pred.argmax(dim=1).cpu())
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    train_trues = np.hstack(train_trues)
    train_preds = np.hstack(train_preds)
    train_acc = skmetrics.accuracy_score(y_true=train_trues, y_pred=train_preds)
    train_f1_score = skmetrics.f1_score(train_trues,train_preds,average="macro")

    #验证模型
    model.eval()
    validate_loss = []
    validate_trues = []
    validate_preds = []
    with torch.no_grad(): #不计算梯度，加快运算速度
        for batch_idx, (X, y) in enumerate(validate_dataloader):
            X, y = X.to(device), y.reshape(-1,).to(device, dtype=torch.long)
            pred = model(X)
            loss = criterion(pred, y)
            validate_trues.append(y.cpu())
            validate_preds.append(pred.argmax(dim=1).cpu())
            validate_loss.append(loss.item())
    
    validate_trues = np.hstack(validate_trues)
    validate_preds = np.hstack(validate_preds)
    validate_acc = skmetrics.accuracy_score(y_true=validate_trues, y_pred=validate_preds)
    validate_f1_score = skmetrics.f1_score(validate_trues,validate_preds,average="macro")

    end = timeit.default_timer()        
    
    print(f"[epoch: {epoch_idx+1:3}/{n_epochs:3}] || train_loss:{np.sum(train_loss):6.2f} || train_acc:{train_acc*100:5.2f}% || train_mf1:{train_f1_score:4.2f}\
    || val_loss:{np.sum(validate_loss):6.2f} || val_acc:{validate_acc*100:5.2f}% || val_mf1:{validate_f1_score:4.2f} ({end-start:4.2f}s)")

    if best_acc < validate_acc:
        best_acc = validate_acc
        print(f"[epoch: {epoch_idx+1:3}/{n_epochs:3}] save best model...")
        torch.save(model.state_dict(), "./models/model_GRU.pt")

#     wandb.log({"train_loss": np.sum(train_loss), 
#                "train_acc": train_acc,
#                "val_loss": np.sum(validate_loss),
#                "val_acc": validate_acc})

# wandb.finish()
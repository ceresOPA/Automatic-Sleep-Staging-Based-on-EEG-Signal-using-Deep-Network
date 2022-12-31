import numpy as np
import torch

from network import SleepNet

import argparse

def predict(seq_len=16, input_file="./data/eeg_data.txt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载模型
    model = SleepNet(seq_len=seq_len, network="GRU")
    model.load_state_dict(torch.load("./models/model_GRU.pt", map_location=device))
    model.to(device)
    # 加载数据，这里只提供了简单的txt文件加载，输入大小为1维序列数据，要更换npy或其他文件也并不复杂
    # 训练数据的每个信号片段是30s，采样频率100Hz，30*100 = 3000
    signal = np.loadtxt(input_file)
    batch_size = len(signal)//seq_len//3000
    signal = signal[:batch_size*seq_len*3000].reshape(-1, 1, 3000) # [batch*seq_len, channel, signal]
    # 模型预测
    X = torch.from_numpy(signal).to(device, dtype=torch.float)
    preds = model(X)
    preds = preds.argmax(dim=1).cpu()

    return preds

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--input_file", type=str, default="./data/eeg_signal.txt")
    args = parser.parse_args()

    preds = predict(args.seq_len, input_file=args.input_file)

    # 绘制睡眠分期图像
    plt.plot(preds)
    plt.title("Automatic Sleep Staging")
    plt.xlabel("Epochs")
    plt.ylabel("Sleep Stages")
    plt.yticks(ticks=[0, 1, 2, 3, 4], labels=["Wake", "N1", "N2", "N3", "REM"])
    plt.show()
# 基于单通道脑电信号的自动睡眠分期研究
**Automatic Sleep Staging Based on EEG Signal using Deep Network**

> 本项目参考自https://github.com/flower-kyo/Tinysleepnet-pytorch



## 简介

本项目使用了Sleep-EDF公开数据集的SC数据进行实验，一共153条整晚的睡眠记录，使用Fpz-Cz通道，采样频率为100Hz，数据获取与具体数据信息可前往数据集官网查看：[Sleep-EDF Database Expanded v1.0.0 (physionet.org)](https://physionet.org/content/sleep-edfx/1.0.0/)。

整套代码写的较为简洁，而且有添加相应的注释，因此进行分享，而且不仅仅说是睡眠分期，也可以作为学习如何使用神经网络去进行**时序数据分类**问题的一个入门项目，包括怎么用GRU、LSTM和Attention这些经典网络结构。

网络结构（具体可查看network.py文件）：

- 网络整体结构类似于[TinySleepNet](https://ieeexplore.ieee.org/document/7961240/)，对RNN部分进行了修改，增加了双向RNN、GRU、Attention等网络结构，可根据参数进行调整选择。

- 定义了seq_len参数，可以更灵活地调整batch_size与seq_len。

数据集加载（具体可查看dataset.py文件）

- 直接继承自torch的Dataset，并定义了seq_len和shuffle_seed，方便调整输入，并复现实验。

训练（具体可查看train.py文件）：

- 定义并使用了focal loss损失函数
- 在实验中有使用wandb，感觉用起来还挺方便的，非常便于实验记录追溯

测试（具体可查看test.py文件）：

- 可以输出accuracy、mf1、recall_confusion_matrics、precision_confusion_matrics、f1_confusion_matrics评价指标



## 使用

### 1. 安装环境

```
pip install -r requirements.txt
```

### 2. 数据准备

下载数据，信号和标注文件格式为edf文件。

```
python downloading_sleepedf.py
```

处理数据，并保存为numpy数组。

```
python prepare_data.py
```

### 3. 模型训练

```
python train.py --n_epochs 150 --batch_size 16 --seq_len 64 --network "GRU"
```

如果不知道要使用哪些哪些参数，可以看下代码，或者使用`python train.py -h`来查看。

### 4. 模型测试

```
python test.py --batch_size 16 --seq_len 64 --network "GRU" --model_path "./models/model_GRU.pt" --data_path "./data/sleepedf/npz"
```

可输出precision、recall和F1的混淆矩阵。

### 5. 模型预测

上传了一个小seq_len训练的模型，效果没那么好，但可以拿来测试体验一下。

```
python predict.py --seq_len 16 --input_file "./data/eeg_signal.txt"
```



## 补充

我自己是在服务器上跑的，使用显卡为A100 256G显存，因此seq_len会设置的比较大些，自己可根据具体情况进行调整。有发现seq_len越大，效果会稍微好些，在设置seq_len=64，使用GRU时，测试集上准确率可达87%，MF1可达0.80。但使用Attention或者Transformer的Encoder进行训练时，发现效果并不怎么好，只能达到72%左右，而且如果堆叠太多层encoder就会跑不起来了，不是很懂。
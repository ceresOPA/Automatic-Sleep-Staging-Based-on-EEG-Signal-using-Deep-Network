# 参考自(reference) https://github.com/flower-kyo/Tinysleepnet-pytorch/blob/main/network.py
# 网络上增加了双向RNN配置，并测试使用了GRU和Attention
# 因为重写了数据集加载的代码，代码更加简洁，可灵活调整输入序列长度seq_len

import torch
import torch.nn as nn
from collections import OrderedDict

class SleepNet(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(SleepNet, self).__init__()
        self.padding_edf = {  
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        first_filter_size = int(100 / 2.0)  
        first_filter_stride = int(100 / 16.0) 
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )
            # 上面这段CNN的代码，是完全使用了参考代码的网络结构
        if network == "LSTM":
            self.rnn = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        elif network == "GRU":
            self.rnn = nn.GRU(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        elif network == "Attention":
            # encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
            # 这里其实用transformer_encoder也可以，但是如果叠加太多层，会跑不起来，而且发现使用attention的效果很一般
            self.qtrans = nn.Sequential(nn.Linear(2048,256), nn.ReLU(inplace=True))
            self.ktrans = nn.Sequential(nn.Linear(2048,256), nn.ReLU(inplace=True))
            self.vtrans = nn.Sequential(nn.Linear(2048,256), nn.ReLU(inplace=True)) 
            self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4,  batch_first=True)
        
        self.rnn_dropout = nn.Dropout(p=0.5)
        if network != "Attention": 
            # 使用双向RNN的时候，会有前向和后向的两个输出进行拼接作为最后的输出，因此大小需要×2
            self.fc = nn.Linear(hidden_size*2, 5) if is_bidirectional else nn.Linear(hidden_size, 5)
        else:
            self.fc = nn.Linear(256, 5)

        self.is_bidirectional = is_bidirectional
        self.hidden_size = hidden_size
        self.network = network
        self.seq_len = seq_len

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)
        x = self.cnn(x)
        x = x.reshape(-1, self.seq_len, 2048)  # batch first == True
        assert x.shape[-1] == 2048
        if self.network != "Attention":
            x, _ = self.rnn(x)
            x = x.reshape(-1, self.hidden_size*2) if self.is_bidirectional else x.reshape(-1, self.hidden_size)
        else:
            # attention除了输出x外，还输出了attn_weights，也就是query和key运算后经类softmax的score，多头的话会进行平均
            x, _ = self.attention(self.qtrans(x),self.ktrans(x),self.vtrans(x))
            x = x.reshape(-1, 256)
        x = self.rnn_dropout(x)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    from torchsummaryX import summary

    model = SleepNet(network="GRU", seq_len=32)
    # torch可以自己计算出隐单元输入的大小，这里就不自己预先定义了
    # state = (torch.zeros(size=(1, 15, 128)),
    #          torch.zeros(size=(1, 15, 128)))
    # torch
    # state = (state[0].to(self.device), state[1].to(self.device))
    summary(model, torch.randn(size=(20*32, 1, 3000))) # batch_size = 20, seq_len = 32

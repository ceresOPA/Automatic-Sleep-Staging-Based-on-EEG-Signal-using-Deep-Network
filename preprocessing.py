
# 参考自(reference) https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py

import pyedflib
import numpy as np
import matplotlib.pyplot as plt

stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "MOVE": 5,
    "UNK": 6,
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}

def edf2numpy(signal_file_path, annotation_file_path, select_channels):
    """
    读取edf文件, 预处理并转换为narray格式的数据
    """
    # 读取edf文件
    X_data = pyedflib.EdfReader(signal_file_path)
    y = pyedflib.EdfReader(annotation_file_path)
    # 看标注文件和信号文件的起始时间点是否一致
    assert X_data.getStartdatetime() == y.getStartdatetime()
    epoch_duration = X_data.datarecord_duration
    # 正常划分的信号均为30s一个片段
    if X_data.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
        epoch_duration = epoch_duration / 2
    # 可以读取的通道 ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']
    channels = X_data.getSignalLabels()
    assert select_channels not in channels
    # 按理说这里应该是采用频率，但是得到的是采样数，采样频率100Hz，30s，3000个点
    n_epoch_samples = int(X_data.getSampleFrequency(channels.index(select_channels[0]))/X_data.datarecord_duration*epoch_duration)
    signals = X_data.readSignal(channels.index(select_channels[0])).reshape(-1, n_epoch_samples)
    # 读取选择的通道的信号
    for select_channel_idx in range(1,len(select_channels)):
        n_epoch_samples = int(X_data.getSampleFrequency(channels.index(select_channels[select_channel_idx]))/X_data.datarecord_duration*epoch_duration)
        signal = X_data.readSignal(channels.index(select_channels[select_channel_idx])).reshape(-1, n_epoch_samples)
        signals = np.append(signals, signal, axis=0)

    # 读取标签
    labels = []
    total_duration = 0
    ann_onsets, ann_durations, ann_stages = y.readAnnotations()
    for a in range(len(ann_stages)):
        onset_sec = int(ann_onsets[a])
        duration_sec = int(ann_durations[a])
        ann_str = "".join(ann_stages[a])

        # Sanity check
        # 有一条记录是按60s划分的，将其拆开，片段变为原来的2倍
        n_epochs = X_data.datarecords_in_file
        if X_data.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert signals.shape[0] == n_epochs, f"signal: {signals.shape} != {n_epochs}"
        
        # onset为当前偏移值, total_duration是累计量 
        assert onset_sec == total_duration

        # 获取标签值，转换为AASM标准
        label = ann2label[ann_str]

        # Compute # of epoch for this stage
        # epoch_duration是30s，正常情况下duration_sec一定是30的倍数，如果非倍数则报错
        if duration_sec % epoch_duration != 0:
            print(f"Something wrong: {duration_sec} {epoch_duration}")
            raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
        duration_epoch = int(duration_sec / epoch_duration)
        # Generate sleep stage labels
        label_epoch = np.ones(duration_epoch, dtype=np.int) * label
        labels.append(label_epoch)

        total_duration += duration_sec

        print("Include onset:{}, duration:{}, label:{} ({})".format(
            onset_sec, duration_sec, label, ann_str
        ))
    labels = np.hstack(labels)

    # Remove annotations that are longer than the recorded signals
    # 去掉多出来的标签，因为没有与之对应的信号
    labels = labels[:signals.shape[0]]

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)

    # Select only sleep periods
    w_edge_mins = 30
    # 找到不为Wake期的第一个下标
    nw_idx = np.where(y != stage_dict["W"])[0]
    # 保留前后30个片段
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx+1)
    print("Data before selection: {}, {}".format(x.shape, y.shape))
    x = x[select_idx]
    y = y[select_idx]
    print("Data after selection: {}, {}".format(x.shape, y.shape))

    # Remove movement and unknown
    move_idx = np.where(y == stage_dict["MOVE"])[0]
    unk_idx = np.where(y == stage_dict["UNK"])[0]
    if len(move_idx) > 0 or len(unk_idx) > 0:
        remove_idx = np.union1d(move_idx, unk_idx)
        print("Remove irrelavant stages")
        print("  Movement: ({}) {}".format(len(move_idx), move_idx))
        print("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
        print("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
        print("  Data before removal: {}, {}".format(x.shape, y.shape))
        select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        x = x[select_idx]
        y = y[select_idx]
        print("  Data after removal: {}, {}".format(x.shape, y.shape))

    return x, y


if __name__ == "__main__":
    signal_file_path = "./data/sleepedf/sleep-cassette/SC4001E0-PSG.edf"
    annotation_file_path = "./data/sleepedf/sleep-cassette/SC4001EC-Hypnogram.edf"

    x, y = edf2numpy(signal_file_path, annotation_file_path, select_channels=["EEG Fpz-Cz"])
    print(x.shape, y.shape)

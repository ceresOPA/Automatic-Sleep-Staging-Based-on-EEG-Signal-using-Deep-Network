import os
import glob
import numpy as np

from preprocessing import edf2numpy

file_path = "./data/sleepedf/sleep-cassette/"
output_path = "./data/sleepedf/npz"

if not os.path.exists(output_path):
    os.makedirs(output_path)

signal_file_list = glob.glob(os.path.join(file_path, "*PSG.edf"))
annotation_file_list = glob.glob(os.path.join(file_path, "*Hypnogram.edf"))

for signal_file in signal_file_list:
    subject_code = signal_file.split('/')[-1].split('-')[0][:-2]
    annotation_file = [item_file for item_file in annotation_file_list if subject_code in item_file][0]
    print(subject_code)
    print(signal_file)
    print(annotation_file)
    x, y = edf2numpy(signal_file, annotation_file, ["EEG Fpz-Cz"])
    print("----------------------------------", x.shape, y.shape, "-----------------------------")
    np.savez(os.path.join(output_path,f"{subject_code}.npz"), x=x, y=y)
import os
import hashlib
import wget

# 数据集下载地址
sleepedf_url = "https://www.physionet.org/files/sleep-edfx/1.0.0"
output_dir = os.path.join("./data", "sleepedf")
record_file = os.path.join(output_dir, "sleepedf_records.txt")

# 输出目录 ./data/sleepedf
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# SHA256SUMS.txt中记录着每条数据的链接与SHA256码
if os.path.exists(record_file):
    os.remove(record_file)
url = sleepedf_url + "/" + "SHA256SUMS.txt"
wget.download(url, record_file)

download_files = []
with open(record_file) as f:
    for l in f.readlines():
        l = l.strip()

        tmp = l.split(" ")
        sha256hash = tmp[0]
        fname = tmp[-1]

        if 'sleep-cassette' in fname:
            download_url = sleepedf_url + "/" + fname
            save_f = os.path.join(output_dir, fname)
            save_dir = os.path.dirname(save_f)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not os.path.isfile(save_f):
                print(f"\nDownloading {download_url} to {save_f}")
                wget.download(download_url, save_f)
            
            # Check SHA256 校验文件，若码相同，则说明文件未被修改，若不同，则说明文件被修改过
            with open(save_f, "rb") as ff:
                b = ff.read()
                readable_hash = hashlib.sha256(b).hexdigest()
                assert sha256hash == readable_hash

                print(f'Downloaded {save_f}')
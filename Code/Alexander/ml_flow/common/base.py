import os
import tarfile
from six.moves import urllib
import numpy as np
import pandas as pd
import hashlib


# 获取数据
def fetch_data_with_url(data_url: str, save_path: str, save_name: str):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    tgz_path = os.path.join(save_path, save_name)
    urllib.request.urlretrieve(data_url, tgz_path)  # 下载
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path=save_path)  # 解压
    data_tgz.close()


# ==========================切分测试集==========================
def split_train_test(data: pd.DataFrame, test_ratio: float):
    shuffled_indices = np.random.permutation(len(data))  # 打乱顺序
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    print(test_indices)
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# 根据字段获取哈希值
def test_set_check(identifier, test_ratio: float, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# 根据哈希值保留固定测试集
def split_train_test_by_id(data: pd.DataFrame, test_ratio: float, id_column: str, hash=hashlib.md5):
    ids = data[id_column]
    # 返回bool list
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

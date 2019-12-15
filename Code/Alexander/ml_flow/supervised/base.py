import pandas as pd


# ==========================切分特征和标签==========================
def split_features_and_labels(train_set: pd.DataFrame, label_name: str):
    # drop()创建了一份数据的副本
    features = train_set.drop(label_name, axis=1)
    labels = train_set[label_name].copy()
    return features, labels


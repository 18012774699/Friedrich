import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array
from scipy import sparse
from ml_flow.custom_class.transform import DataFrameSelector
from ml_flow.custom_class.transform import CustomizedTransform
from ml_flow.custom_class.encoder import CategoricalEncoder


# ====================数据清洗====================
# 处理特征丢失的问题(数字)
def fill_missing_num(features: pd.DataFrame, fill_strategy: str, text_column: list):
    imputer = Imputer(strategy=fill_strategy)
    # 此处只能处理数值属性，创建一份不包含文本属性的skip_text_data的数据副本
    num_features = features.drop(text_column, axis=1)  # axis=1: 按标签
    imputer.fit(num_features)

    # print(imputer.statistics_)
    # print(num_features.median().values)

    X = imputer.transform(num_features)  # numpy数组
    num_features_transform = pd.DataFrame(X, columns=num_features.columns)  # DataFrame
    return num_features_transform


# 处理文本和分类属性
def processing_text_and_classification_attributes(features: pd.DataFrame, text_column: list):
    text_features = features[text_column]

    # encoder = LabelEncoder()
    # cat_encoded = encoder.fit_transform(text_features)
    # print(text_features)
    # print(cat_encoded)
    # print(encoder.classes_)

    # encoder = OneHotEncoder()
    # text_features_one_hot = encoder.fit_transform(cat_encoded.reshape(-1, 1))
    # print(text_features_one_hot)

    # 独热编码(One-Hot Encoding)
    # sparse_output=True，就可以得到一个稀疏矩阵
    encoder = LabelBinarizer(sparse_output=True)
    text_features_one_hot = encoder.fit_transform(text_features)
    return text_features_one_hot


# 数据清洗
def data_cleaning(features: pd.DataFrame, fill_strategy: str, text_column: list):
    # 1)处理特征丢失的问题(数字)
    num_features_transform = fill_missing_num(features, fill_strategy, text_column)

    # 2)处理文本和分类属性，独热编码(One-Hot Encoding)
    text_features_one_hot = processing_text_and_classification_attributes(features, text_column)
    return num_features_transform, text_features_one_hot


# ========================转换Pipeline========================
# 数据清洗、处理文本和分类属性、自定义操作、特征缩放
def create_features_transform_pipeline(features: pd.DataFrame, fill_strategy: str, text_column: list,
                                       scale_type: str = "normalization",
                                       customized_transform=CustomizedTransform()):
    # 此处只能处理数值属性，创建一份不包含文本属性的skip_text_data的数据副本
    num_features = list(features.drop(text_column, axis=1))  # axis=1: 按标签
    scaler = StandardScaler() if scale_type == "std_scaler" else MinMaxScaler()
    num_pipeline = Pipeline([('selector', DataFrameSelector(num_features)),
                             ('imputer', Imputer(strategy=fill_strategy)),  # 处理特征丢失的问题
                             ('customized', customized_transform),  # 自定义操作
                             ('scaler', scaler)])  # 特征缩放

    # 处理文本和分类属性，独热编码(One-Hot Encoding)
    cat_pipeline = Pipeline([('selector', DataFrameSelector(text_column)),
                             ('label_binarizer', CategoricalEncoder(encoding="onehot-dense"))])

    full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                                   ("cat_pipeline", cat_pipeline)])
    return full_pipeline

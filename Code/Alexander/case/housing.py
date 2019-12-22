import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer
from ml_flow.common import base as cb
from ml_flow.supervised import base as sb
from ml_flow.common import pipeline as cp
from ml_flow.supervised import model_selection as sm

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "../datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# 预览数据
def preview_data(housing):
    # 表头
    print(housing.head())
    # 表信息描述
    print(housing.info())
    # 数值分类，以及其计数
    print(housing["ocean_proximity"].value_counts())
    # DataFrame统计信息
    print(housing.describe())

    # 绘制统计图
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()


# 切分测试集
def split_test_set(housing):
    # train_set, test_set = base.split_train_test(housing, 0.2)
    # print(test_set.info())

    housing_with_id = housing.reset_index()  # adds an `index` column
    train_set, test_set = cb.split_train_test_by_id(housing_with_id, 0.2, "index")

    '''保留固定的测试集'''
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = cb.split_train_test_by_id(housing_with_id, 0.2, "id")
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    '''假设专家告诉你，收入中位数是预测房价中位数非常重要的属性。采用分层采样偏差。'''
    # ceil: 向上取整
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # train_index: ndarray, stratified_train_set: df
    # split: 分层切分类，split.split()返回generator
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        stratified_train_set = housing.loc[train_index]
        stratified_test_set = housing.loc[test_index]

    print(housing["income_cat"].value_counts() / len(housing))

    # 删除income_cat属性，使数据回到初始状态：
    for set in (stratified_train_set, stratified_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    return stratified_train_set, stratified_test_set


# 数据可视化
def data_visualization(housing):
    # housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 # 每个圈的半径表示分区的人口（选项s），颜色代表价格（选项c）。
                 # 我们用预先定义的颜色图（选项cmap）jet，它的范围是从蓝色（低价）到红色（高价）
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()


# 查找关联性
def search_for_correlation(housing):
    # 使用corr()方法计算出每对属性间的标准相关系数（也称作皮尔逊相关系数）
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # 绘制每个数值属性对其他数值属性的图
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

    # 最有希望用来预测房价中位数的属性是收入中位数，因此将这张图放大
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()

    # 属性组合试验
    # 加工数据，查找关联
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)


# 此处为添加自定义组合属性，添加rooms_per_household、population_per_household、bedrooms_per_room
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


if __name__ == '__main__':
    # 获取数据的函数, 一次
    # cb.fetch_data_with_url(HOUSING_URL, HOUSING_PATH, "housing.csv")

    housing = pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv"))

    # 预览数据
    preview_data(housing)

    # 切分测试集
    strat_train_set, strat_test_set = split_test_set(housing)

    # 数据探索
    # housing = strat_train_set.copy()
    # data_visualization(housing)
    # search_for_correlation(housing)

    # 切分特征和标签
    housing, housing_labels = sb.split_features_and_labels(strat_train_set, label_name="median_house_value")

    # 转换Pipeline: 数据清洗、处理文本和分类属性、自定义操作、特征缩放(标准化)
    # 自定义添加rooms_per_household、population_per_household、bedrooms_per_room
    full_pipeline = cp.create_features_transform_pipeline(housing, fill_strategy="median",
                                                          text_column=["ocean_proximity"], scale_type="std_scaler",
                                                          customized_transform=CombinedAttributesAdder())

    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared.shape)

    # 选择并训练模型
    # 模型性能、准确性、过拟合
    # 训练集的评分比验证集的评分低很多，表示模型过拟合
    # sm.choose_model(housing_prepared, housing_labels)

    # 模型微调
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = sm.fine_tune_model(housing_prepared, housing_labels, forest_reg, param_grid)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    cv_res = grid_search.cv_results_
    for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
        print(np.sqrt(-mean_score), params)

    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing["ocean_proximity"])
    feature_importances = grid_search.best_estimator_.feature_importances_
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_one_hot_attribs = list(encoder.classes_)
    attributes = list(housing.drop("ocean_proximity", axis=1)) + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))

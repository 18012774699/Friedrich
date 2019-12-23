import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class MyCrossValidation:
    def __init__(self, data_prepared, labels, scoring: str = "neg_mean_squared_error", cv: int = 10):
        self.data_prepared = data_prepared
        self.labels = labels
        self.scoring = scoring
        self.cv = cv

    # Scikit - Learn交叉验证功能期望的是效用函数（越大越好）而不是成本函数（越低越好），因此得分函数实际上与MSE相反（即负值）
    # 因此前面加上负号
    # 过拟合：训练集的评分比验证集的评分低很多，表示模型过拟合
    # 模型性能：RMSE小
    # 准确性：标准差小
    def display_evaluation_indicator(self, regression, print_type: str):
        regression.fit(self.data_prepared, self.labels)
        predictions = regression.predict(self.data_prepared)
        mse = mean_squared_error(self.labels, predictions)  # 计算MSE
        rmse = np.sqrt(mse)
        print(print_type, ":")
        print('Train set rmse: ', rmse)

        # K折交叉验证
        scores = cross_val_score(regression, self.data_prepared, self.labels, scoring=self.scoring, cv=self.cv)
        rmse_scores = np.sqrt(-scores)
        # print("Scores:", rmse_scores)
        print("Validation set rmse mean:", rmse_scores.mean())
        print("Standard deviation:", rmse_scores.std())


# =============================选择模型(线性或非线性)=============================
def choose_model(data_prepared, labels, scoring: str = "neg_mean_squared_error", cv: int = 10):
    print("=========================choose model=========================")
    cross_validation = MyCrossValidation(data_prepared, labels, scoring, cv)
    # 尝试线性回归
    lin_reg = LinearRegression()
    cross_validation.display_evaluation_indicator(lin_reg, "lin_reg")

    # 尝试决策树，非线性关系
    tree_reg = DecisionTreeRegressor()
    cross_validation.display_evaluation_indicator(tree_reg, "tree_reg")

    # 尝试随机森林
    forest_reg = RandomForestRegressor()
    cross_validation.display_evaluation_indicator(forest_reg, "forest_reg")


# =============================模型微调=============================
def fine_tune_model(data_prepared, labels, regression, param, mode: str = 'randomized', n_iter: int = 100):
    # n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    # 如果GridSearchCV是以（默认值）refit=True开始运行的，则一旦用交叉验证找到了最佳的估计量，就会在整个训练集上重新训练
    if mode == 'randomized':
        model_selection = RandomizedSearchCV(regression, param, cv=5, scoring='neg_mean_squared_error', n_iter=n_iter)
    else:
        model_selection = GridSearchCV(regression, param, cv=5, scoring='neg_mean_squared_error')
    model_selection.fit(data_prepared, labels)
    return model_selection
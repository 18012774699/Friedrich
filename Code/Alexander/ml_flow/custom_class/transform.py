from sklearn.base import BaseEstimator, TransformerMixin


# 自定义转换量
# 自定义转换量，操作过程：先实例化、fit(param)、transform(param)
class CustomizedTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

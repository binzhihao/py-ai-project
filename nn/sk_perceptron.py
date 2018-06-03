import pandas as pd
import numpy as np
import data.iris.loader as loader
from sklearn.linear_model import Perceptron


if __name__ == '__main__':
    x, y = loader.load_iris()

    # 探索数据
    print(x.head())
    print('species: ', pd.unique(y.species))

    # 花瓣长度，宽度
    x_train = x.loc[:, ['petal_length', 'petal_width']]
    # 转成0-1二分类
    y_train = (y.species == 'setosa').astype(np.int)

    per_clf = Perceptron(random_state=1)
    per_clf.fit(x_train, y_train)

    y_predict = per_clf.predict([[3.5, 0.2]])
    print(y_predict)

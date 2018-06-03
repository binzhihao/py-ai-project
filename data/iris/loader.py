import pandas as pd
import os

module_path = os.path.dirname(__file__)


def load_iris(path=module_path):
    iris = pd.read_csv(os.path.join(path, 'iris.csv'))
    x = iris.iloc[:, :-1]
    y = iris.iloc[:, [-1]]
    return x, y

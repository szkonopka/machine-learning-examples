import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Perceptron import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
X = df.iloc[0:100, [0, 2]].values
Y = df.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)

exec(open('scripts/PreparePlot.py').read())

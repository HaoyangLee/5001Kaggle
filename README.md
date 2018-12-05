# 5001Kaggle
There are three .py file: GBDT.py (use GradientBoostingRegressor), lgbm.py (use lightgbm) and NN.py (use Neural Network). The GBDT.py get the best result.

1. Programming languages: Python 3.6.1

2. Required packages:

import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from scipy.stats import pearsonr

from keras.layers import Dense, Activation

from keras.models import Sequential

3. How to run: 
Change the file path of "train.csv" and "test.csv", then run GBDT.py, lgbm.py and NN.py seperately. The GBDT.py get the best result.

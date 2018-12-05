
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from keras.layers import Dense, Activation
from keras.models import Sequential

# function encode() is used for doing the LabelEncode and OneHotEncode on the data
def encode(data):
    lab_en = LabelEncoder().fit(data)
    lab_en_data = lab_en.transform(data)

    onehot_en = OneHotEncoder(sparse=False).fit(lab_en_data.reshape(-1, 1))
    onehot_data = onehot_en.transform(lab_en_data.reshape(-1, 1))

    return onehot_data


def main():
    raw_train = pd.read_csv("E:/HKUST/5001_Data_Analytics/kaggle/data1028/train.csv")
    raw_test = pd.read_csv("E:/HKUST/5001_Data_Analytics/kaggle/data1028/test.csv")
    labels = raw_train['time']

    raw = raw_train.append(raw_test)
    raw.drop('id',axis=1, inplace=True)

    penalty = encode(raw['penalty'].values)
    # Select some useful features
    fea = ['l1_ratio', 'alpha', 'max_iter', 'random_state','n_jobs', 'n_samples', 'n_features',
           'n_classes', 'n_clusters_per_class', 'n_informative', 'flip_y', 'scale']

    # Construct polynomial features based on the original features
    poly = PolynomialFeatures()
    poly_fea = poly.fit_transform(raw[fea])
    poly_fea_df = pd.DataFrame(dict(zip(poly.get_feature_names(),np.transpose(poly_fea))))

    # Select features according to their Pearson correlation coefficients
    R = []
    P = []
    poly_fea_df_fea_name = poly_fea_df.columns.values.tolist()
    for col_name in poly_fea_df_fea_name:
        r, p = pearsonr(poly_fea_df[col_name].iloc[:raw_train.shape[0]], labels)
        R.append(r)
        P.append(p)

    # Rank the features based on their Pearson correlation coefficients
    d = dict(zip(poly_fea_df_fea_name, R))
    d_sorted = sorted(d.items(), key=lambda item: abs(item[1]), reverse=True)
    pearsonr_fea = []

    for i in range(50):
        pearsonr_fea.append(d_sorted[i][0])

    piersen = pearsonr_fea.pop(0)
    fea_stand = StandardScaler().fit_transform(poly_fea_df[pearsonr_fea])

    # Select features according to their variances：var_thresh = p(1-p)
    # VT = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # fea_stand = VT.fit_transform(fea_stand)

    all_fea = np.hstack((penalty, fea_stand))

    # split the whole dataset into train_fea data and test_fea data
    # and then split the train_fea into training data and testing data
    # test_fea data is used for prediction
    train_fea = all_fea[:raw_train.shape[0]]
    test_fea = all_fea[raw_train.shape[0]:]

    # Select features according to chi2 distribution
    # train_fea = SelectKBest(chi2, k=2).fit_transform(abs(train_fea), labels)


    model = Sequential()
    model.add(Dense(64, input_dim=53, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mse', optimizer='sgd')
    model.fit(train_fea, labels, epochs=1000, batch_size=400, verbose=0)
    model.evaluate(train_fea,labels)
    y_predict = model.predict(test_fea)
    y_predict_out = pd.DataFrame(y_predict)
    y_predict_out.to_csv("NNsubmission.csv", index_label='Id', header = ['time'])

if __name__ == '__main__':
    main()
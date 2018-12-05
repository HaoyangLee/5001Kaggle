import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
# from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2



# function encode() is used for doing the LabelEncode and OneHotEncode on the data
def encode(data):
    lab_en = LabelEncoder().fit(data)
    lab_en_data = lab_en.transform(data)

    onehot_en = OneHotEncoder(sparse=False).fit(lab_en_data.reshape(-1, 1))
    onehot_data = onehot_en.transform(lab_en_data.reshape(-1, 1))

    return onehot_data

def main():
    raw_train = pd.read_csv("./train.csv")
    raw_test = pd.read_csv("./test.csv")
    labels = raw_train['time']

    raw = raw_train.append(raw_test)
    raw.drop('id',axis=1, inplace=True)

    # Construct new features
    raw['sam_fea'] = raw['n_samples'] * raw['n_features']
    raw['cla_clu'] = raw['n_classes'] * raw['n_clusters_per_class']
    penalty = encode(raw['penalty'].values)
    # Select some useful features
    fea = ['l1_ratio', 'alpha', 'max_iter', 'n_jobs', 'n_samples', 'n_features',
           'n_classes', 'n_clusters_per_class', 'n_informative', 'flip_y','scale','sam_fea','cla_clu']

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
    for i in range(90):
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

    X_train, X_test, y_train, y_test = train_test_split(train_fea, labels, test_size=0.2, random_state=0)

    # grid search to find the best parameters of the model
    # param_grid = [
    #     # to avoid taking too much time for you to test my code, I put some optional values of
    #     # parameters in the comment and put validated "good" parameters in the code.
    #     #{'n_estimators': [20, 100, 500], 'max_depth': [5, 10]}
    # ]

    gdbt = GradientBoostingRegressor(n_estimators = 500, max_depth=5, random_state=1)
    gdbt.fit(X_train, y_train)
    MSE = mean_squared_error(y_test, gdbt.predict(X_test))
    print("MSE:",MSE)

    y_predict = gdbt.predict(test_fea)
    y_predict_out = pd.DataFrame(y_predict)
    y_predict_out[y_predict_out < 0] = 0
    y_predict_out.to_csv("mysubmission.csv", index_label='Id', header = ['time'])

if __name__ == '__main__':
    main()
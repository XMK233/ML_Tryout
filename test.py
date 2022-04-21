import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import pandas as pd
import numpy as np
import tqdm, datetime, pickle
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ngboost as ngb

from sklearn import datasets

import plotnine
from plotnine import *

import seaborn as sns

import os, time

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten

data_train = pd.read_csv("preprocessedDataset/pre1.csv")
numerical_category_fewValues = [
    "homeOwnership",
    "verificationStatus",
    "initialListStatus",
    "applicationType",
    "n11",
    "n12",
]
numerical_category_manyValues = [
    "regionCode",
    "employmentTitle",
    "purpose",
    "postCode",
    "title",
]
date_type = [
    'issueDate',
    'earliesCreditLine',
    'issueDateDT',
    'earliesCreditLineDT',
    'earliesCreditLineYear',
    'earliesCreditLineMonth',
    'issueYear',
    'issueMonth'
]
numerical_serial = [
    "loanAmnt","interestRate","installment","annualIncome","dti","delinquency_2years","ficoRangeLow","ficoRangeHigh","openAcc",
    "pubRec","pubRecBankruptcies","revolBal","revolUtil","totalAcc","n0","n1","n2","n3",
    "n4","n5","n6","n7","n8","n9","n10","n13","n14",
    "term",
]
object_serial = [
    "grade",
    "subGrade",
    "employmentLength"
]

features = [f for f in data_train.columns if f not in ['id', 'isDefault', "policyCode", "issueDate", "earliesCreditLine"] and '_outliers' not in f]
y_full = data_train['isDefault']
x_full = data_train[features]

from sklearn.model_selection import train_test_split

X_, X_test, y_, y_test = train_test_split(x_full, y_full, test_size=0.5, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_, y_, test_size=0.6, random_state=0)
del X_, y_

from sklearn.model_selection import train_test_split


def save_dmatrix_1(all_feas, feas_cols_v2, target_label, dmtrx_fname, dmtrx_dir=""):
    '''
    保存的dmatrix的名字可以自定义。
    '''
    from sklearn.model_selection import train_test_split
    y = all_feas[target_label]
    train, test = train_test_split(all_feas, test_size=0.25, random_state=30, shuffle=True, stratify=y)

    import xgboost as xgb

    all_train_matrix = xgb.DMatrix(
        train[feas_cols_v2].values,
        train[target_label].values,
        feature_names=feas_cols_v2
    )
    all_train_matrix.save_binary(
        os.path.join(dmtrx_dir, f"{dmtrx_fname}--train")
    )  # (f'train_matrix_dongzhi_v1-{target_label}')

    all_test_matrix = xgb.DMatrix(
        test[feas_cols_v2].values,
        test[target_label].values,
        feature_names=feas_cols_v2
    )
    all_test_matrix.save_binary(
        os.path.join(dmtrx_dir, f"{dmtrx_fname}--test")
    )  # (f'test_matrix_dongzhi_v1-{target_label}')

    del train, all_train_matrix, test, all_test_matrix
    import gc
    gc.collect()

    return


def save_dmatrix(all_feas, feas_cols_v2, target_label):
    from sklearn.model_selection import train_test_split
    y = all_feas[target_label]
    train, test = train_test_split(all_feas, test_size=0.25, random_state=30, shuffle=True, stratify=y)

    import xgboost as xgb

    all_train_matrix = xgb.DMatrix(
        train[feas_cols_v2].values,
        train[target_label].values,
        feature_names=feas_cols_v2
    )
    all_train_matrix.save_binary(f'train_matrix_dongzhi_v1')  # (f'train_matrix_dongzhi_v1-{target_label}')

    all_test_matrix = xgb.DMatrix(
        test[feas_cols_v2].values,
        test[target_label].values,
        feature_names=feas_cols_v2
    )
    all_test_matrix.save_binary(f'test_matrix_dongzhi_v1')  # (f'test_matrix_dongzhi_v1-{target_label}')

    del train, all_train_matrix, test, all_test_matrix
    import gc
    gc.collect()

    return


def train_model_with_different_label(target_label):
    dtrain = xgb.DMatrix(data=f'train_matrix_dongzhi_v1')  # (data=f'train_matrix_dongzhi_v1-{target_label}')
    dtest = xgb.DMatrix(data=f'test_matrix_dongzhi_v1')  # (data=f'test_matrix_dongzhi_v1-{target_label}')

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'scale_pos_weight': 3,
        'learning_rate': 0.1,
        'reg_lambda': 5,
        'reg_alpha': 0,
        'colsample_bytree': 0.8,
        # 'tree_method': 'gpu_hist', # 'gpu_exact',
        #             "gpu_id":7,
    }
    params['nthread'] = 25
    train_start_time = time.time()
    booster_maidian = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dtrain, 'train'), (dtest, 'test')],
                                early_stopping_rounds=50, verbose_eval=10)
    train_end_time = time.time()

    print(f"train time: {train_end_time - train_start_time}")

    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    return booster_maidian


def train_model_with_different_label_1(data_dir, data_name):
    dtrain = xgb.DMatrix(
        data=os.path.join(data_dir, f'{data_name}--train')
    )
    dtest = xgb.DMatrix(
        data=os.path.join(data_dir, f'{data_name}--test')
    )

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'scale_pos_weight': 3,
        'learning_rate': 0.1,
        'reg_lambda': 5,
        'reg_alpha': 0,
        'colsample_bytree': 0.8,
        #             'tree_method': 'gpu_exact',
        #             "gpu_id": 5,
    }
    params['nthread'] = 25
    train_start_time = time.time()
    booster_maidian = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dtrain, 'train'), (dtest, 'test')],
                                early_stopping_rounds=50, verbose_eval=10)
    train_end_time = time.time()

    print(f"train time: {train_end_time - train_start_time}")

    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    return booster_maidian


## 可调参数区
model_dir = "./trainedModel/"
model_name = "xgb.json"
feas_cols_v2 = X_train.columns
feas_discard_num = 5 ## 每次砍掉尾部这么多个特征再拟合模型得到AUC
num_iter = 3 ## 砍尾多少次
all_feas = data_train
target_label = "isDefault"


xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(model_dir, model_name))

iptc_scores = xgb_model.get_score(importance_type="total_gain")
importances = []
for i in range(len(iptc_scores)):
    importances.append((feas_cols_v2[i], iptc_scores[f"f{i}"]))
importances.sort(key=lambda x: x[1], reverse=True)
sorted_features = [_[0] for _ in importances]

# for ni in tqdm.tqdm(range(1, num_iter + 1)):
#     save_dmatrix_1(
#         all_feas,
#         feas_cols_v2[:-1*(ni)*feas_discard_num],
#         target_label,
#         dmtrx_fname = f"feas-{len(feas_cols_v2) - (ni)*feas_discard_num}",
#         dmtrx_dir = "preprocessedDataset/"
#     )

for ni in tqdm.tqdm(range(1, num_iter + 1)):
    dt_name = f"feas-{len(feas_cols_v2) - (ni) * feas_discard_num}"
    #     model = train_model_with_different_label_1(
    #         data_dir = "preprocessedDataset/",
    #         data_name = dt_name
    #     )

    data_dir = "preprocessedDataset/"
    data_name = dt_name

    dtrain = xgb.DMatrix(
        data=os.path.join(data_dir, f'{data_name}--train')
    )
    dtest = xgb.DMatrix(
        data=os.path.join(data_dir, f'{data_name}--test')
    )

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'scale_pos_weight': 3,
        'learning_rate': 0.1,
        'reg_lambda': 5,
        'reg_alpha': 0,
        'colsample_bytree': 0.8,
    }
    # params['nthread'] = 25
    train_start_time = time.time()
    booster_maidian = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=10
    )
    train_end_time = time.time()

    model = booster_maidian
    model.save_model(f'trained_models/{dt_name}')
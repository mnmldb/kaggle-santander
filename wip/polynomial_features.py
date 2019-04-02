'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Kernel: Santander Magic LGB 0.901
URL: https://www.kaggle.com/jesucristo/santander-magic-lgb-0-901
+
Add PolynomialFeatures
URL: http://tekenuko.hatenablog.com/entry/2016/09/19/193520
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numba import jit
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
import seaborn as sns

import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))

#additional package
from sklearn.preprocessing import PolynomialFeatures

#load data
train_df = pd.read_csv('../input/train.csv') # (200000, 202)
test_df = pd.read_csv('../input/test.csv') # (200000, 201)

train_features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
test_features = [c for c in test_df.columns if c not in ['ID_code']]

target = train_df['target']

#add polynomial features
quadratic = PolynomialFeatures(degree = 2) #instance to create quadratic features
quadratic_train = quadratic.fit_transform(train_df[train_features].values) # (200000, 20301) 20301 = 200C2 + 200 + 200
quadratic_test = quadratic.fit_transform(test_df[test_features].values)

q_train_df = pd.concat(train_df[['ID_code', 'target']], pd.DataFrame(quadratic_train[:,1:]), axis=1) # delete first column of quadratic_train
q_test_df = pd.concat(test_df['ID_code'], pd.DataFrame(quadratic_test[:,1:]), axis=1) # delete first column of quadratic_test


#Clasification augment
@jit
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

#Run model
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}

#kfold = 15
#folds = StratifiedKFold(n_splits=kfold, shuffle=False, random_state=44000)
num_folds = 11
features = [c for c in q_train_df.columns if c not in ['ID_code', 'target']]

folds = KFold(n_splits=num_folds, random_state=44000)
oof = np.zeros(len(q_train_df)) #oof: out of fold
getVal = np.zeros(len(q_train_df))
predictions = np.zeros(len(target))
feature_importance_df = pd.DataFrame()


print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(q_train_df.values, target.values)):
    
    X_train, y_train = q_train_df.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = q_train_df.iloc[val_idx][features], target.iloc[val_idx]
    
    X_tr, y_tr = augment(X_train.values, y_train.values)
    X_tr = pd.DataFrame(X_tr)
    
    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    val_data = lgb.Dataset(q_train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(q_train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    getVal[val_idx]+= clf.predict(q_train_df.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits
    predictions += clf.predict(q_test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print("\n >> CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

#Submission
submission = pd.DataFrame({"ID_code": q_test_df.ID_code.values})
submission["target"] = predictions
submission.to_csv("submission.csv", index=False)


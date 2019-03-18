#import libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier,Pool
from IPython.display import display
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import norm
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import time
import glob
import sys
import os
import gc

from sklearn.metrics import roc_auc_score
from reduce_mem_usage import reduce_mem_usage

#import dataset
train= pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')

#reduce memory
train, NAlist = reduce_mem_usage(train)
test, NAlist = reduce_mem_usage(test)

#set fold
fold_n=5
folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)

#prepare X, y
cols=["target", "ID_code"]
X = train.drop(cols, axis=1)
y = train["target"]

X_test = test.drop('ID_code', axis=1)

#lightgbm
#params is based on following kernel https://www.kaggle.com/brandenkmurray/nothing-works
params = {'objective' : "binary", 
               'boost':"gbdt",
               'metric':"auc",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.01,
               'num_leaves' : 13,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.05,
               'bagging_freq' : 5,
               'bagging_fraction' : 0.4,
               'min_data_in_leaf' : 80,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : 1}

## %%time
y_pred_lgb = np.zeros(len(X_test))
num_round = 1000000
for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    lgb_model = lgb.train(params,train_data,num_round,#change 20 to 2000
                    valid_sets = [train_data, valid_data],verbose_eval=1000,early_stopping_rounds = 3500)##change 10 to 200
            
    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5


#random forest
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
y_pred_rfc = rfc_model.predict(X_test)

#decision tree
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
y_pred_tree = tree_model.predict(X_test)

#cat boost
train_pool = Pool(train_X,train_y)
cat_model = CatBoostClassifier(
                               iterations=3000,# change 25 to 3000 to get best performance 
                               learning_rate=0.03,
                               objective="Logloss",
                               eval_metric='AUC',
                              )
cat_model.fit(train_X,train_y,silent=True)
y_pred_cat = cat_model.predict(X_test)

#result
print('LightGBM: {}'.format(roc_auc_score(y, y_pred_lgb)))
print('Random Forest: {}'.format(roc_auc_score(y, y_pred_rfc)))
print('Decision Tree: {}'.format(roc_auc_score(y, y_pred_tree)))
print('Cat Boost: {}'.format(roc_auc_score(y, y_pred_cat)))




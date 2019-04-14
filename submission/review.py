'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---Reference---

Name: List of Fake Samples and Public/Private LB split
URL: https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split

Name: Santander Magic LGB 0.901
URL: https://www.kaggle.com/jesucristo/santander-magic-lgb-0-901
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# LOAD LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np, gc
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
import statsmodels.api as sm
import tqdm

# GET INDICIES OF REAL TEST DATA FOR FE
test_path = '../input/test.csv'

df_test = pd.read_csv(test_path)
df_test.drop(['ID_code'], axis=1, inplace=True)
df_test = df_test.values

unique_samples = []
unique_count = np.zeros_like(df_test)
for feature in range(df_test.shape[1]):
    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

print('Found',len(real_samples_indexes),'real test')
print('Found',len(synthetic_samples_indexes),'fake test')

###################

d = {}
for i in range(200): d['var_'+str(i)] = 'float32'
d['target'] = 'uint8'
d['ID_code'] = 'object'

train = pd.read_csv('../input/train.csv', dtype=d)
test = pd.read_csv('../input/test.csv', dtype=d)

print('Loaded',len(train),'rows of train')
print('Loaded',len(test),'rows of test')

# FREQUENCY ENCODE
def encode_FE(df, col, test):
    cv = df[col].value_counts()
    nm = col + '_FE'
    df[nm] = df[col].map(cv)
    test[nm] = test[col].map(cv)
    test[nm].fillna(0, inplace=True)
    if cv.max()<=255:
        df[nm] = df[nm].astype('uint8')
        test[nm] = test[nm].astype('uint8')
    else:
        df[nm] = df[nm].astype('uint16')
        test[nm] = test[nm].astype('uint16')        
    return

test['target'] = -1
comb = pd.concat([train,test.loc[real_samples_indexes]],axis=0,sort=True) #combine training data and real samples test data -> 300,000 rows
for i in range(200):
    encode_FE(comb,'var_'+str(i),test)

train = comb[:len(train)]; del comb
print('Added 200 new magic features!')


#LGBM param
param = {
    'learning_rate': 0.04,
    'num_leaves': 3,
    'metric':'auc',
    'boost_from_average':'false',
    'feature_fraction': 1.0,
    'max_depth': -1,
    'objective': 'binary',
    'verbosity': -10}

num_folds = 5
features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']

folds = StratifiedKFold(n_splits=num_folds, random_state=40000)
oof = np.zeros(len(train)) #oof: out of fold
#getVal = np.zeros(len(train))
predictions = np.zeros(len(target))
feature_importance_df = pd.DataFrame()

print('Light GBM Model')
for fold_, (trn_index, val_index) in enumerate(folds.split(train.values, target.values)):
    X_train, y_train = train.iloc[trn_index][features], target.iloc[trn_index] #prepare training set
    X_val, y_val = train.iloc[val_index][features], target.iloc[val_index] #prepare validation set
    
    print("Fold idx:{}".format(fold_ + 1))
    #create data set in the fold
    trn_data = lgb.Dataset(X_train, label=y_train) #training set
    val_data = lgb.Dataset(X_val, label=y_val) #validation set
    
    #create and train model
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 3000)
    #valid_setsにtrn_dataとval_dataのリストを渡すことで、それぞれのaucがされる
    
    #predict oof
    oof[val_index] = clf.predict(X_val, num_iteration=clf.best_iteration) #最もよかったモデルで予測
    
    #predict test (has to be divided by the number of total folds in order to make average)
    predictions += clf.predict(test[features],num_iteration=clf.best_iteration) / folds.n_splits
    
    #summarize feature importance in the fold
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    
    #add feature importance in the fold
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

#check CV score
print("\n >> CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

#Submission
submission = pd.DataFrame({"ID_code": test.ID_code.values})
submission["target"] = predictions
submission.to_csv("submission.csv", index=False)
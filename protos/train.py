'''
実行時はprotosディレクトリにおいてpython train.pyとする
'''

import pandas as pd
import numpy as np
import pickle #複数のオブジェクトを1つのまとまりに保存し、後で読み込める
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc

#load_data.pyより必要な関数のみを読み込む
from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

#ログの結果を吐き出す場所
DIR = 'result_tmp/'

#sample submission fileの場所
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'


#コマンドラインから呼ばれた場合の処理 (importされたときに自動で実行されない)
if __name__ == '__main__':

    ###ログの設定
    #標準出力にはINFOレベル以上を流す
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    #ファイル出力にはDEBUGレベル以上を流す
    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    #-----train-----#
    df = load_train_data()
    #ID_codeも除外する必要がある->後ほどの.sort_values('ID_code')はできなくなる
    x_train = df.drop(['ID_code', 'target'], axis=1)
    y_train = df['target'].values #valuesでpandas->numpy配列へ

    #train dataとtest dataで列の順番が違う可能性があるため、
    #後ほどtest dataをtrain dataの列の並びに合わせる
    use_cols = x_train.columns.values

    #ログ
    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('data preparation end {}'.format(x_train.shape))

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)

    #ログ
    logger.info('train end')

    #-----test-----#
    df = load_test_data()

    #train dataとtest dataで列の順番が違う可能性があるため、
    #後ほどtest dataをtrain dataの列の並びに合わせる
    use_cols = x_train.columns.values 
    x_test = df[use_cols] #.sort_values('ID_code') #後ほどsampleと合わせるため、並べ替えを行っておく

    #ログ
    logger.info('test data load end{}'.format(x_test.shape))
    pred_test = clf.predict_proba(x_test) #今回は確率で出力

    #-----submit file-----#
    #新しくsubmitファイルを作るわけではなく、kaggleから提供されているsampleファイルを上書きする
    #ファイルの読み込み
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE) #.sort_values('ID_code') #predictと合わせるため、並べ替えを行っておく
    #結果の上書き
    df_submit['target'] = pred_test
    #結果の書き出し
    df_submit.to_csv(DIR + 'submit.csv')

    #ログ
    logger.info('end')


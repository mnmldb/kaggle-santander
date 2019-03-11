'''
実行時はprotosディレクトリにおいてpython train.pyとする
tail -F train.py.logで更新中のログを見れる
'''

import pandas as pd
import numpy as np
import pickle #複数のオブジェクトを1つのまとまりに保存し、後で読み込める
from tqdm import tqdm #処理の進捗状況をプログレスバーとして表示
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


    #-----Cross Varidation-----# (訓練データを複数に分割して、何回か検証を行い、その平均を取って汎化性能とする)
    #StratifiedKFoldが一番いいらしい (各CVのスプリットで0,1を一定の比率にしてくれる)
    #StratifiedKFoldはユーザーごとにスプリットしないといけない場合や、時系列データには使用してはいけない
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) #分割は5はほしい、データ大きい場合は3   必ず乱数は固定する


    #-----Grid Search-----# (パラメータの候補値を総当たりで試し、最適値を出す)
    #パラメータの候補値を辞書に格納
    #モデルによってパラメータは異なる
    all_params = {'C': [10**i for i in range(-3, 4)], #C 正則化パラメータは10**i -> logの数値で探すことが多い
                  'fit_intercept': [True, False],
                  'penalty': ['l2', 'l1'],
                  'random_state': [0]}

    #一番スコアが小さかったとき用の初期値を準備
    min_score = 100
    min_params = None

    #Grid Searchのループ
    #tqdmを使うことで、進捗状況を可視化 (イテレータを返す、引数にリストを使用)
    for params in tqdm(list(ParameterGrid(all_params))): #パラメータの候補値をParameterGridに渡し、ループにする
        #ログ出力
        logger.info('params: {}'.format(params))
        #結果を格納するリスト
        list_auc_score = []
        list_logloss = []

        #Cross Validationのループ
        for train_idx, valid_idx in cv.split(x_train, y_train):
            #cv.splitで列の行のインデックスが返ってくる
            #CVで使用しているのは訓練データのみなので注意 (テストデータは使用しない)
            
            #xの準備
            trn_x = x_train.iloc[train_idx, :] #pandas
            val_x = x_train.iloc[valid_idx, :] #pandas
            #yの準備
            trn_y = y_train[train_idx] #numpy array
            val_y = y_train[valid_idx] #numpy array
            #モデルの訓練
            clf = LogisticRegression(**params) #パラメータを可変長で渡す
            clf.fit(trn_x, trn_y)
            #予測
            pred = clf.predict(val_x)
            #モデルの評価　今回の尺度はloglossとAUCを使用
            sc_logloss = log_loss(val_y, pred) #log loss 小さい方がいい
            sc_auc = (-1) * roc_auc_score(val_y, pred) #AUC　大きいほうがいい (上に合わせるためマイナスをかけている)
            #尺度をリストに格納
            list_logloss.append(sc_logloss)
            list_auc_score.append(sc_auc)

            #ループごとにログ出力 (debug level)
            #メッセージにインデントを入れることでわかりやすくする (tail -F train.py.log で確認するとき)
            logger.debug('  logloss: {}, auc: {}'.format(sc_logloss, sc_auc))
            #break #ここでbreakを入れることでKFold 1回だけでやることも可能 (エラーが起きる場合にこうしてシンプルにしてデバッグする)

        #cross validation後の平均値を格納
        sc_logloss = np.mean(list_logloss) #loglossは小さい方がいい
        sc_auc = np.mean(list_auc_score) #aucは大きいほうがいい (すでにマイナスはかかっている)
        
        logger.info('logloss: {}, auc: {}'.format(sc_logloss, sc_auc))

        #順次モデルの精度を評価し、以前のものより優れていればAUCとパラメータを上書きする
        #今回roglossは使用していない
        if min_score > sc_auc: #マイナスに反転させているため
            min_score = sc_auc
            min_params = params

        #ログ出力
        logger.info('logloss: {}, auc: {}'.format(sc_logloss, sc_auc))
        logger.info('current min score: {}, auc: {}'.format(min_score, sc_auc))

    #ログ出力
    logger.info('minimum params: {}'.format(min_params))
    logger.info('minimum auc: {}'.format(min_score))

    #一番良かったパラメータで最後にすべての訓練データでモデルを訓練
    clf = LogisticRegression(**min_params)
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
    pred_test = clf.predict(x_test) #今回は0 or 1で出力

    #-----submit file-----#
    '''
    #新しくsubmitファイルを作るわけではなく、kaggleから提供されているsampleファイルを上書きする
    #ファイルの読み込み
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE) #.sort_values('ID_code') #predictと合わせるため、並べ替えを行っておく
    #結果の上書き
    df_submit['target'] = pred_test
    #結果の書き出し
    df_submit.to_csv(DIR + 'submit.csv')
    '''
    df_submit = pd.DataFrame()
    df_submit['ID_code'] = df['ID_code'] #test data
    df_submit['target'] = pred_test #pred_testはnumpy arrayだがそのままDataFrameに格納できる

    #結果の書き出し
    df_submit.to_csv(DIR + 'submit.csv', index=False)

    #ログ
    logger.info('end')


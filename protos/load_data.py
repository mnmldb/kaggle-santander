import numpy as np
import pandas as pd
from tqdm import tqdm
from logging import getLogger

TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'


#logをためる場合の定型文
logger = getLogger(__name__)

'''
__name__とは、Pythonのプログラムがどこから呼ばれて実行されているかを格納しているグローバル変数
Pythonのプログラムは、コマンドラインから(Python xxx.py)直接呼ばれるか、
import文で他のプログラムから参照されて呼ばれる
その区別を__name__で行うことができる
・コマンドラインから直接呼ばれた場合、__name__には__main__という文字列が格納される
・importで他のプログラムから呼ばれた場合、ファイル名が格納される
'''

def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    logger.debug('exit')
    return df

def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')
    return df

def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    logger.debug('exit')
    return df

'''
Pythonでは、インポートされた際ファイルの中身は自動で実行される
インポートされた際に自動で実行されないよう、
以下のようにif __name__ = '__main__'というif文を書く
このif文の内部は、importされた際には実行されない
(一方で、コマンドラインから呼ばれた場合は実行される)
'''
if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())

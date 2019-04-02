'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
URL: https://blog.amedama.jp/entry/pandas-group-sampling
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import pandas as pd

def train_sample(df_train, sample_ratio, y_label):
    gdf_train = df_train.groupby(y_label)
    df_train_sample = gdf_train.apply(lambda x: x.sample(n=round(len(x) * sample_ratio)))
    df_train_sample.reset_index(drop=True, inplace=True)
    return df_train_sample
    
def test_sample(df_test, sample_ratio):
    df_test_sample = df_test.sample(n=round(len(df_test) * sample_ratio))
    df_test_sample.reset_index(drop=True, inplace=True)
    return df_test_sample

if __name__ == '__main__':
    train_sample()
    test_sample()

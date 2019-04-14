


df0_metrics = df0[features].describe()
df0_metrics.loc['Skew'] = df1[features].skew().values
df0_metrics.loc['Kurt'] = df1[features].kurt().values


df_metrics = df_train.describe()

df_1_4 = df_metrics.loc['25%'] #Series
df_3_4 = df_metrics.loc['75%'] #Series

df_1_4 = df_1_4[1:] #delete target
df_3_4 = df_3_4[1:] #delete target


#create columns
q1_columns = [i + '_q1' for i in features]
q3_columns = [i + '_q3' for i in features]

#if value is smaller than 25%, return True
q1 = pd.DataFrame(df_1_4.values > df_train[features].values, columns=q1_columns) * 1 #convert True/False to 1/0

#if value is larger than 75%, return True
q3 = pd.DataFrame(df_3_4.values < df_train[features].values, columns=q3_columns) * 1 #convert True/False to 1/0

#concat all dataframes
df_train_all = pd.concat([df_train, q1, q3], axis=1)






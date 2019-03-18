from reduce_mem_usage import reduce_mem_usage
import pandas as pd


train= pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')

train, NAlist = reduce_mem_usage(train)
test, NAlist = reduce_mem_usage(test)

print(train.head())
print(test.head())


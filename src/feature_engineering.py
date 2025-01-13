import pandas as pd 
import numpy as np 

train = pd.read_csv("/home/lelou1505/DS/Titanic/data/raw/train.csv")
test = pd.read_csv("/home/lelou1505/DS/Titanic/data/raw/test.csv")

train['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.nan
all_data = pd.concat([train,test])


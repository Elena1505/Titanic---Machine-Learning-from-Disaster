import pandas as pd 
import numpy as np 
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

# Application of the same processes as in feature engineering, but for all data (train and test)
def process_data_feature(data:DataFrame): 
    data['Cabin_multiple'] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    data['Cabin_adv'] = data.Cabin.apply(lambda x: str(x)[0])
    data['Numeric_ticket'] = data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    data['Ticket_letters'] = data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
    data['Name_title'] = data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    return data 


def process_null_values(data: DataFrame, train: DataFrame): 
    data.Age = data.Age.fillna(train.Age.median())
    data.Fare = data.Fare.fillna(train.Fare.median())
    data.dropna(subset=['Embarked'],inplace = True)
    return data 


def normalize_values(data:DataFrame):
    data['Norm_fare'] = np.log(data.Fare+1)
    return data 


def encode_values(data:DataFrame):
    data.Pclass = data.Pclass.astype(str)
    dummies_data = pd.get_dummies(data[['Pclass','Sex','Age','SibSp','Parch','Norm_fare','Embarked','Cabin_adv','Cabin_multiple','Numeric_ticket','Name_title','train_test']])
    return dummies_data


def scale_data(data:DataFrame): 
    scale = StandardScaler()
    dummies_data_scaled = data.copy()
    dummies_data_scaled[['Age','SibSp','Parch','Norm_fare']]= scale.fit_transform(dummies_data_scaled[['Age','SibSp','Parch','Norm_fare']])
    return dummies_data_scaled


def split_data(dummies_data: DataFrame, data:DataFrame):
    X_train = dummies_data[dummies_data.train_test == 1].drop(['train_test'], axis =1)
    X_test = dummies_data[dummies_data.train_test == 0].drop(['train_test'], axis =1)
    y_train = data[data.train_test==1].Survived
    return X_train, X_test, y_train



import pandas as pd 
from pandas import DataFrame


# Application of the same processes as in feature engineering, but for all data (train and test)
def process_data_feature(data:DataFrame): 
    data['cabin_multiple'] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    data['cabin_adv'] = data.Cabin.apply(lambda x: str(x)[0])
    data['numeric_ticket'] = data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    data['ticket_letters'] = data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
    data['name_title'] = data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    return data 


def process_null_values(data: DataFrame, train: DataFrame): 
    data.Age = data.Age.fillna(train.Age.median())
    data.Fare = data.Fare.fillna(train.Fare.median())
    data.dropna(subset=['Embarked'],inplace = True)
    return data 
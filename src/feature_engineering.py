from src.data_exploration import compare_data
import pandas as pd 
import numpy as np 
from pandas import DataFrame

def combine_data(train_path:str, test_path:str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train['train_test'] = 1
    test['train_test'] = 0
    test['Survived'] = np.nan
    all_data = pd.concat([train,test], ignore_index=True)
    return train, test, all_data


def process_cabin_feature(train:DataFrame):
    # Creates a new column 'Cabin_multiple' indicating the number of cabins allocated to each passenger (0 if no cabins are specified).
    train['Cabin_multiple'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    # Adds a 'Cabin_adv' column containing the first letter of the cabin, representing the deck (or 'n' if the cabin is missing).
    train['Cabin_adv'] = train.Cabin.apply(lambda x: str(x)[0])
    pivot_cabin_multiple = compare_data(train, 'Survived', 'Cabin_multiple', 'Ticket', 'count')
    pivot_cabin_adv = compare_data(train, 'Survived', 'Cabin_adv', 'Ticket', 'count')
    return train, train['Cabin_multiple'].value_counts(), pivot_cabin_multiple, train["Cabin_adv"].value_counts(), pivot_cabin_adv


def process_ticket_feature(train:DataFrame):
    # Adds a 'numeric_ticket' column indicating whether the ticket number is entirely numeric (1) or not (0).
    train['Numeric_ticket'] = train.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    # Adds a 'ticket_letters' column containing the letters or prefixes extracted from the ticket number (normalized to lower case), or 0 if there is no prefix.
    train['Ticket_letters'] = train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
    pivot_numeric_ticket = compare_data(train, 'Survived', 'Numeric_ticket', 'Ticket', 'count')
    pivot_ticket_letter = compare_data(train, 'Survived', 'Ticket_letters', 'Ticket', 'count')
    return train, train['Numeric_ticket'].value_counts(), pivot_numeric_ticket, train['Ticket_letters'].value_counts(), pivot_ticket_letter


def process_name_feature(train:DataFrame):
    # Adds a 'name_title' column containing the title extracted from the name (for example, 'Mr', 'Mrs', 'Miss', etc.).
    train['Name_title'] = train.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    return train, train['Name_title'].value_counts()

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os 
import seaborn as sns 
from typing import List
from pandas import DataFrame


def describe(raw_path:str):
    data = pd.read_csv(raw_path)
    info = data.info()
    description = data.describe()
    return data, info, description


def describe_numerical_data(raw_path:str, column_list:List, figure_path:str):
    data = pd.read_csv(raw_path)
    df_num = data[column_list]

    for i in df_num.columns: 
        plt.hist(df_num[i])
        plt.title(i)
        plt.savefig(os.path.join(figure_path, f"{i}.png"))

    sns.heatmap(df_num.corr())
    plt.savefig(os.path.join(figure_path, "Heatmap"))
    plt.close()


def describe_categorical_data(raw_path:str, column_list:List[str], figure_path:str):
    data = pd.read_csv(raw_path)
    df_cat = data[column_list]

    for i in df_cat.columns:
        value_counts = df_cat[i].value_counts()
        if len(value_counts) > 10:
            value_counts = value_counts[:10]
        sns.barplot(x=value_counts.index, y=value_counts).set_title(i)
        plt.savefig(os.path.join(figure_path, f"{i}.png"))
        plt.close()


def compare_data(data:DataFrame, index:str, column:str, values:str, aggfunc:str):
    return(pd.pivot_table(data, index = index, columns = column, values = values, aggfunc = aggfunc))

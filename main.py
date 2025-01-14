from src.data_exploration import describe, describe_numerical_data, describe_categorical_data, compare_data
from src.feature_engineering import combine_data, process_cabin_feature, process_ticket_feature, process_name_feature

def main(): 
    train_path = "/home/lelou1505/DS/titanic/data/raw/train.csv"
    test_path = "/home/lelou1505/DS/titanic/data/raw/test.csv"
    figure_path = "/home/lelou1505/DS/titanic/reports/figures"

    data, infos, description = describe(train_path)
    #describe_numerical_data(train_path, ['Age','SibSp','Parch','Fare'], figure_path)
    #describe_categorical_data(train_path, ['Survived','Pclass','Sex','Ticket','Cabin','Embarked'], figure_path)
    #print(compare_data(data, "Survived", column="Pclass", values="Ticket", aggfunc="count"))
    #print(compare_data(data, "Survived", column="Sex", values="Ticket", aggfunc="count"))
    #print(compare_data(data, "Survived", column="Embarked", values="Ticket", aggfunc="count"))
    train, test, all_data = combine_data(train_path, test_path)
    train, _, _, _, _ = process_cabin_feature(train)
    train, _, _, _, _ = process_ticket_feature(train)
    train, _ = process_name_feature(train)
    print(train)


if __name__ == "__main__":
    main()